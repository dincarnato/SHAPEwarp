package SW::DB;

use strict;
use Core::Mathematics;
use Core::Utils;
use Fcntl qw(SEEK_SET SEEK_END);
use SW::DB::Entry;

use base qw(Data::IO);

use constant EOF => "\x5b\x65\x6f\x66\x64\x62\x5d";
use constant VERSION => 1;

sub new {

    my $class = shift;
    my %parameters = @_ if (@_);

    my $self = $class->SUPER::new(%parameters);
    $self->_init({ index       => undef,
                   buildIndex  => 0,
                   dbSize      => 0,
                   _offsets    => {},
                   _lengths    => {},
                   _lastOffset => 0 }, \%parameters);

    $self->{binmode} = ":raw";

    $self->_openfh();
    $self->_validate();
    $self->_loadIndex() if ($self->mode() ne "w"); # r | w+

    return($self);

}

sub _validate {

    my $self = shift;

    my ($fh, $eof);
    $fh = $self->{_fh};

    $self->SUPER::_validate();

    if ($self->mode() eq "r") {

        seek($fh, -7, SEEK_END);
        read($fh, $eof, 7);

        $self->throw("Invalid DB file (EOF marker is absent)") unless ($eof eq EOF);

        $self->reset();

    }
    else { $self->throw("DB size must be a positive integer") if (!ispositive($self->{dbSize})); }

    $self->{index} = $self->{file} . ".dbi" if ($self->{buildIndex} &&
                                                defined $self->{file} &&
                                                !defined $self->{index});

}

sub _loadIndex {

    my $self = shift;

    my ($fh, $n, @n);
    $fh = $self->{_fh};

    if (-e $self->{index}) {

        my ($data, $idlen, $id, $offset);

        open(my $ih, "<:raw", $self->{index}) or $self->throw("Unable to read from DBI index file (" . $! . ")");
        while(!eof($ih)) {

            read($ih, $data, 4);
            $idlen = unpack("L<", $data);

            read($ih, $data, $idlen);
            $id = substr($data, 0, -1); # Removes the "\x00" string terminator

            read($ih, $data, 8);
            $offset = unpack("Q<", $data);
            $self->{_offsets}->{$id} = $offset; # Stores the offset for ID

            # Validates offset
            seek($fh, $offset + 4, SEEK_SET);
            read($fh, $data, $idlen);

            $self->throw("Invalid offset in DBI index file for transcript \"" . $id . "\"") if (substr($data, 0, -1) ne $id);

            if ($self->mode() eq "w+") {

                read($fh, $data, 4);
                $self->{_lengths}->{$id} = unpack("L<", $data);

            }

        }
        close($ih);

    }
    else { # Builds missing index

        my ($data, $offset, $idlen, $id,
            $length);
        $offset = 0;

        while($offset < (-s $self->{file}) - 17) { # While the 8 + 2 + 7 bytes of total mapped reads + RC version + EOF Marker are reached

            read($fh, $data, 4);
            $idlen = unpack("L<", $data);

            read($fh, $data, $idlen);
            $id = substr($data, 0, -1); # Removes the "\x00" string terminator

            read($fh, $data, 4);
            $length = unpack("L<", $data);

            $self->{_offsets}->{$id} = $offset;
            $self->{_lengths}->{$id} = $length if ($self->mode() eq "w+");

            $offset += 8 * ($length + 1) + length($id) + 1 + ($length + ($length % 2)) / 2;

            seek($fh, $offset, SEEK_SET);

        }

        if ($self->{buildIndex} &&
            defined $self->{index}) {

            open(my $ih, ">:raw", $self->{index}) or $self->throw("Unable to write DBI index file (" . $! . ")");
            select((select($ih), $|=1)[0]);

            foreach my $id (keys %{$self->{_offsets}}) {

                print $ih pack("L<", length($id) + 1) .                 # len_transcript_id (uint32_t)
                          $id . "\0" .                                  # transcript_id (char[len_transcript_id])
                          pack("Q<", $self->{_offsets}->{$id});         # offset in count table (uint64_t)

            }

            close($ih);

        }

    }

    seek($fh, -17, SEEK_END);
    read($fh, $n, 8);

    # Unpack the 64bit int
    $self->dbSize(unpack("Q<", $n));

    $self->reset();

}

sub read {

    my $self = shift;
    my $seqid = shift if (@_);

    my ($fh, $data, $idlen, $id,
        $length, $sequence, $entry, $eightbytes,
        @reactivity);

    $self->throw("Filehandle isn't in read mode") unless ($self->mode() eq "r");

    $fh = $self->{_fh};

    if (defined $seqid) {

        if (exists $self->{_offsets}->{$seqid}) {

            seek($fh, $self->{_offsets}->{$seqid}, SEEK_SET);

            # Re-build the prev array to allow ->back() call after seeking to a specific sequence
            my @prev = grep {$_ <= $self->{_offsets}->{$seqid}} sort {$a <=> $b} values %{$self->{_offsets}};
            $self->{_prev} = \@prev;

        }
        else { return; }

    }

    # Checks whether RTCEOF marker has been reached
    read($fh, $eightbytes, 17);
    seek($fh, tell($fh) - 17, SEEK_SET);

    # The first 4 bytes are the total size in nt of the db
    if (substr($eightbytes, -7) eq EOF) {

        $self->reset() if ($self->{autoreset});

        return;

    }

    push(@{$self->{_prev}}, tell($fh)) if (!defined $seqid);

    read($fh, $data, 4); #read id length
    $idlen = unpack("L<", $data);

    read($fh, $data, $idlen);
    $id = substr($data, 0, -1);

    read($fh, $data, 4);
    $length = unpack("L<", $data);

    read($fh, $data, ($length + ($length % 2)) / 2);
    $data = unpack("H*", $data);
    $data =~ tr/43210/NTGCA/;
    $sequence = substr($data, 0, $length);

    read($fh, $data, 8 * $length);
    @reactivity = unpack("d*", $data);
    @reactivity = map { isnegative($_) ? "NaN" : $_ } @reactivity;

    $entry = SW::DB::Entry->new( id         => $id,
                                 sequence   => $sequence,
                                 reactivity => \@reactivity );

    return($entry);

}

sub write {

    my $self = shift;
    my @entries = @_ if (@_);

    my ($fh, @offsets);
    $fh = $self->{_fh};

    $self->throw("Filehandle isn't in write\/append mode") unless ($self->mode() =~ m/^w/);

    foreach my $entry (@entries) {

        if (!blessed($entry) ||
            !$entry->isa("SW::DB::Entry")) {

            $self->warn("Method requires a valid SW::DB::Entry object");

            next;

        }

        my ($id, $seq, $length, @reactivity);
        $id = $entry->id();
        $seq = $entry->sequence;
        $seq =~ tr/NTGCA/43210/;
        $length = length($seq);
        @reactivity = map { isnan($_) ? -999 : $_ } $entry->reactivity();

        if (!defined $seq ||
            !defined $id ||
            !@reactivity) {

            $self->warn("Empty DWS::DB::Entry object");

            next;

        }

        if ($self->mode() eq "w+") {

            # An undefined issue exists with Perl's threads, that sometimes causes
            # failure in index loading, thus we make a check and reload the index (just in case)
            if (!keys %{$self->{_offsets}}) { $self->_loadIndex(); }

            $self->throw("Sequence ID \"" . $id . "\" is absent in file \"" . $self->{file} . "\"") if (!exists $self->{_lengths}->{$id});
            $self->throw("Sequence \"" . $id . "\" length differs from sequence in file \"" . $self->{file} . "\"") if ($length != $self->{_lengths}->{$id});

            seek($fh, $self->{_offsets}->{$id}, SEEK_SET);

        }

        print $fh pack("L<", length($id) + 1) .             # len_transcript_id (uint32_t)
                  $id . "\0" .                              # transcript_id (char[len_transcript_id])
                  pack("L<", length($seq)) .                # len_seq (uint32_t)
                  pack("H*", $seq) .                        # seq (uint8_t[(len_seq+1)/2])
                  pack("d*", @reactivity);

        $self->{dbSize} += $length;

        if ($self->{buildIndex} &&
            $self->{mode} ne "w+") {

            $self->{_offsets}->{$id} = $self->{_lastOffset};
            push(@offsets, $self->{_lastOffset});
            $self->{_lastOffset} += 8 * ($length + 1) + length($id) + 1 + ($length + ($length % 2)) / 2;

        }

    }

    return(wantarray() ? @offsets : \@offsets) if ($self->{buildIndex});

}

sub dbSize {

    my $self = shift;
    my $n = shift if (@_);

    $self->throw("DB size must be a positive integer") if (defined $n &&
                                                           !ispositive($n));

    $self->{dbSize} = $n if (defined $n);

    return($self->{dbSize});

}

sub close {

    my $self = shift;

    if ($self->mode() =~ m/^w/) {

        my $fh = $self->{_fh};

        seek($fh, -17, SEEK_END) if ($self->mode() eq "w+");

        print $fh pack("Q<", $self->{dbSize}) .  # Total mapped reads (64bit int)
                  pack("S<", VERSION) .
                  EOF; # EOF Marker

        if ($self->{buildIndex} &&
            defined $self->{index}) {

            open(my $ih, ">:raw", $self->{index}) or $self->throw("Unable to write DBI index (" . $! . ")");
            select((select($ih), $|=1)[0]);

            foreach my $id (sort {$self->{_offsets}->{$a} <=> $self->{_offsets}->{$b}} keys %{$self->{_offsets}}) {

                print $ih pack("L<", length($id) + 1) .                 # len_transcript_id (uint32_t)
                          $id . "\0" .                                  # transcript_id (char[len_transcript_id])
                          pack("Q<", $self->{_offsets}->{$id});         # offset in count table (uint64_t)

            }

            close($ih);

        }

    }

    $self->SUPER::close();

}

sub ids {

    my $self = shift;

    my @ids = sort keys %{$self->{_offsets}};

    return(wantarray() ? @ids : \@ids);

}

1;
