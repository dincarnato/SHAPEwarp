package SW::Utils;

use strict;
use List::Util;

use Core::Mathematics qw(:all);
use Core::Utils;
use Interface::ViennaRNA;

use base qw(Exporter);

use constant PI        => 4 * atan2(1, 1);
use constant MINWINLEN => 50;

our @EXPORT = qw(matchScore blockShuffle nullModelStats calcPvalue
                 unpairedProbs);

sub matchScore {

    my ($react1, $react2, $params) = @_;

    $params = checkparameters({ match         => [-0.5, 2],
                                mismatch      => [-6, -0.5],
                                maxReactivity => 1.0 }, $params);

    return($params->{match}->[0]) if (!isnumeric($react1, $react2));

    my ($score, $diff);
    $diff = $react1 >= 1 && $react2 >= 1 ? 0 : abs($react1 - $react2);
    $score = $diff < 0.5 ? maprange(-0.5, 0, @{$params->{match}}, -$diff) : -maprange(0.5, $params->{maxReactivity}, reverse(map { abs($_) } @{$params->{mismatch}}), $diff);

    return($score);

}

sub unpairedProbs {

    my $sequence = shift;
    my $reactivity = shift;
    my $params = shift || {};

    $params = checkparameters({ maxBPspan     => 600,
                                noLonelyPairs => 0,
                                noClosingGU   => 0,
                                slope         => 1.8,
                                intercept     => -0.6,
                                temperature   => 37,
                                winSize       => 800,
                                offset        => 200,
                                winTrim       => 50 }, $params);

    my ($nWins, $length, $interface, @windows,
        @unpairedProb, %bpProbs);
    $length = length($sequence);
    $nWins = max(1, POSIX::floor(($length - $params->{winSize}) / $params->{offset}) + 1);

    # Define windows to be analyzed
    # Windows are defined as: [start, end, 5'-trim, 3'-trim]
    if ($nWins > 1) {

        for (50, 100) {

            my $end = $params->{winSize} - $_ - 1;

            last if ($end < MINWINLEN - 1);

            push(@windows, [0, $end, 0, $params->{winTrim}]);

        }

        for (0 .. $nWins - 1) {

            my ($start, $end);
            $start = $params->{offset} * $_;
            $end = $_ == $nWins - 1 ? $length - 1 : $start + $params->{winSize} - 1;
            push(@windows, [$start, $end, $params->{winTrim}, $params->{winTrim}]);

        }

        for (50, 100) {

            my $start = $length - $params->{winSize} + $_;

            last if ($start > $length - MINWINLEN);

            push(@windows, [$start, $length - 1, $params->{winTrim}, 0]);

        }

    }
    else { push(@windows, [0, $length - 1, 0, 0]); }

    # This fixes the cases in which the first or last windows are < MINWINLEN
    $windows[0]->[2] = 0;
    $windows[-1]->[3] = 0;
    $interface = Interface::ViennaRNA->new();

    for (@windows) {

        my ($start, $end, $trim5, $trim3,
            $winSeq, $fold, @winReact, %winProbs);
        ($start, $end, $trim5, $trim3) = @{$_};

        $winSeq = substr($sequence, $start, $end - $start + 1);
        @winReact = @{$reactivity}[$start .. $end];
        $fold = $interface->fold($winSeq, { reactivity        => \@winReact,
                                            partitionFunction => 1,
                                            %{$params} });
        %winProbs = $fold->bpprobability();
        %bpProbs = _mergeBpHashes(\%bpProbs, \%winProbs, $start);

    }

    @unpairedProb = map { sprintf("%.3f", 1 - (exists $bpProbs{$_} ? min(1, sum(map { mean(@{$_}) } values %{$bpProbs{$_}})) : 0)) } 0 .. $length - 1;

    return(@unpairedProb);

}

sub blockShuffle {

    # data -> reactivities
    # sequence -> RNA sequence (optional)
    # blockSize -> size of the block to be shuffled
    # chunkSize -> if provided, only a chunk of data will be returned

    my %params = @_;

    my ($inBlockShuffle, $blockSize, $chunkSize, $starti,
        $length, @sequence, @data, @dataChunks, @seqChunks);
    @data = @{$params{data}};
    $length = @data;
    $inBlockShuffle = $params{inBlockShuffle};
    $blockSize = $params{blockSize} || 10;
    $chunkSize = $params{chunkSize} || @data;
    @sequence = split("", $params{sequence});

    if ($blockSize == 1 || $length < $blockSize) {

        my @i = List::Util::shuffle(0 .. $#data);
        return(defined $params{sequence} ? ([@data[@i]], join("", @sequence[@i])) : [@data[@i]]);

    }

    $starti = int(rand(min($blockSize, $length - $blockSize)));

    for(my $i = $starti; $i < scalar(@data) - $blockSize + 1; $i += $blockSize) {

        my (@dataChunk, @seqChunk, @i);
        @dataChunk = @data[$i .. $i + $blockSize - 1];
        @seqChunk = @sequence[$i .. $i + $blockSize - 1] if (defined $params{sequence});
        @i = $inBlockShuffle ? List::Util::shuffle(0 .. $#dataChunk) : 0 .. $#dataChunk;

        @dataChunk = @dataChunk[@i];
        @seqChunk = @seqChunk[@i] if (defined $params{sequence});

        if ($i + $blockSize >= scalar(@data) - $blockSize + 1) {

            my (@lastDataChunk, @lastSeqChunk, @i);
            @lastDataChunk = @data[$i + $blockSize .. $#data];
            @lastSeqChunk = @sequence[$i + $blockSize .. $#sequence] if (defined $params{sequence});

            if ($starti) {

                push(@lastDataChunk, @data[0 .. $starti - 1]);
                push(@lastSeqChunk, @sequence[0 .. $starti - 1]) if (defined $params{sequence});

            }

            @i = $inBlockShuffle ? List::Util::shuffle(0 .. $#lastDataChunk) : 0 .. $#lastDataChunk;
            @lastDataChunk = @lastDataChunk[@i];
            @lastSeqChunk = @lastSeqChunk[@i] if (defined $params{sequence});

            push(@dataChunks, \@lastDataChunk);
            push(@seqChunks, \@lastSeqChunk) if (defined $params{sequence});

        }

        push(@dataChunks, \@dataChunk);
        push(@seqChunks, \@seqChunk) if (defined $params{sequence});

    }

    my @i = List::Util::shuffle(0 .. $#dataChunks);
    @dataChunks = map { @{$_} } @dataChunks[@i];
    @dataChunks = @dataChunks[0 .. min($chunkSize, $#dataChunks)];

    if (defined $params{sequence}) {

        @sequence = map { @{$_} } @seqChunks[@i];
        @sequence = @sequence[0 .. min($chunkSize, $#sequence)];

    }

    return(defined $params{sequence} ? (\@dataChunks, join("", @sequence)) : \@dataChunks);

}

sub nullModelStats {

    my @scores = @{$_[0]};
    my $maxIter = $_[1];

    my ($mean, $stdev);

    foreach my $iter (1 .. $maxIter) {

        my (@zscores);
        $mean = mean(@scores);
        $stdev = stdev(@scores);

        last if (!$stdev);

        @zscores = map { ($_ - $mean) / $stdev } @scores;
        @scores = map { $scores[$_] } grep { $zscores[$_] >= -3 && $zscores[$_] <= 5 } 0 .. $#zscores;

    }

    $mean = mean(@scores);
    $stdev = stdev(@scores);

    return($mean, $stdev, \@scores);

}

sub calcPvalue {

    my ($realScore, $mean, $stdev) = @_;

    my ($zscore, $pvalue, $eulerConst);
    $eulerConst = 0.5772156649;
    $zscore = ($realScore - $mean) / $stdev;
    $pvalue = 1 - exp(-exp(-($zscore * PI) / sqrt(6) - $eulerConst));

    return($pvalue);

}

sub _mergeBpHashes {

    my %hash1 = %{$_[0]};
    my %hash2 = %{$_[1]};
    my $start = $_[2];

    my (%merge);

    foreach my $key (keys %hash1) {

        if (ref($hash1{$key}) eq "HASH") {

            my $h2 = exists $hash2{$key - $start} ? $hash2{$key - $start} : {};
            %{$merge{$key}} = _mergeBpHashes($hash1{$key}, $h2, $start);

        }
        else {

            $merge{$key} = $hash1{$key};
            push(@{$merge{$key}}, $hash2{$key - $start}) if (exists $hash2{$key - $start});

        }

        delete($hash2{$key - $start});

    }

    foreach my $key (keys %hash2) {

        if (ref($hash2{$key}) eq "HASH") { $merge{$key + $start} = { map { ($_ + $start) => [ $hash2{$key}->{$_} ] } keys %{$hash2{$key}} }; }
        else { $merge{$key + $start} = [ $hash2{$key} ]; }

    }

    return(%merge);

}

1;
