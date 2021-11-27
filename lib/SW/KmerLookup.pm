package SW::KmerLookup;

use strict;
use Core::Utils;
use Core::Mathematics qw(:all);
use FFI::Platypus 1.00;

our $VERSION = '0.1.0';

my $ffi = FFI::Platypus->new( api => 1, lang => 'Rust' );
$ffi->bundle('SW::KmerLookup');

$ffi->type( 'object(SW::KmerLookup)' => 'KmerLookup' );
$ffi->type( 'object(SW::KmerLookupBuilder)' => 'KmerLookupBuilder' );
$ffi->type( 'object(SW::KmerLookupOkErr)' => 'KmerLookupOkErr' );
$ffi->type( 'object(SW::KmerLookupResults)' => 'KmerLookupResults' );

$ffi->mangler(sub {
  my($name) = @_;
  "kmer_lookup_$name";
});

use base qw(Core::Base);

sub new {

    my $class = shift;
    my %parameters = @_ if (@_);

    my ($self, $builder);
    $self = $class->SUPER::new(%parameters);
    $self->_init({ offset            => 1,
                   matchSequence     => 0,
                   maxSeqDist        => 0,
                   matchGCcontent    => 0,
                   maxGCdiff         => undef,
                   kmerLen           => 12,
                   minKmers          => 2,
                   maxKmerDist       => 10,
                   threads           => 1,
                   minComplexity     => 0.2,
                   maxReactivity     => 1.5,
                   maxMatchesEveryNt => 0 }, \%parameters);

    $self->_validate();
    $builder = SW::KmerLookupBuilder->new($self->{kmerLen});

    if ($self->{matchGCcontent}) {

        $self->{maxGCdiff} = 0.2676 * exp(-0.053 * $self->{kmerLen}) if (!defined $self->{maxGCdiff}); # equation of the line estimated from Rfam alignments
        $builder->max_gc_diff($self->{maxGCdiff});

    }

    $builder->max_sequence_distance($self->{maxSeqDist}) if ($self->{matchSequence});
    $builder->kmer_step($self->{offset});
    $builder->min_kmers($self->{minKmers});
    $builder->max_kmer_merge_distance($self->{maxKmerDist});
    $builder->threads($self->{threads});
    $builder->max_reactivity($self->{maxReactivity});
    $builder->min_complexity($self->{minComplexity});
    $builder->max_matches_every_nt($self->{maxMatchesEveryNt}) if ($self->{maxMatchesEveryNt});

    return($builder->build());
}

sub _validate {

    my $self = shift;

    $self->throw("matchGCcontent must be BOOL") if (!isbool($self->{matchGCcontent}));
    $self->throw("matchSequence must be BOOL") if (!isbool($self->{matchSequence}));
    $self->throw("kmerLen must be a positive INT >= 6") if (!isint($self->{kmerLen}) || $self->{kmerLen} < 6);
    $self->throw("offset must be a positive INT >= 1 and <= kmerLen") if (!isint($self->{offset}) || !inrange($self->{offset}, [1, $self->{kmerLen}]));
    $self->throw("minKmers must be a positive INT >= 1") if (!isint($self->{minKmers}) || $self->{minKmers} < 1);
    $self->throw("maxKmerDist must be a positive INT") if (!isint($self->{maxKmerDist}) || !ispositive($self->{maxKmerDist}));
    $self->throw("maxGCdiff must be comprised between 0 and 1") if (defined $self->{maxGCdiff} && !inrange($self->{maxGCdiff}, [0, 1]));
    $self->throw("maxReactivity must be positive") if (!ispositive($self->{maxReactivity}));
    $self->throw("minComplexity must be positive") if (!ispositive($self->{minComplexity}));
    $self->throw("maxMatchesEveryNt must be a positive INT") if (!ispositive($self->{maxMatchesEveryNt}) || !isint($self->{maxMatchesEveryNt}));
    $self->throw("threads must be a positive INT >= 1") if (!isint($self->{threads}) || $self->{threads} < 1);

}

$ffi->attach( run => ['KmerLookup', 'string', 'f64[]', 'usize', 'string'] => 'KmerLookupOkErr' => sub {
    my ($xsub, $kmer_lookup, $db_path, $query, $query_sequence) = @_;

    my $query_len = @$query;
    $query_sequence = "N" x $query_len if (!defined $query_sequence);

    my $result = $xsub->($kmer_lookup, $db_path, $query, $query_len, $query_sequence);

    if ($result->is_ok()) {
        my @out;
        my $results = $result->get_ok();
        my $results_len = $results->len();
        my $result_index = 0;
        while ($result_index < $results_len) {
            my $dbId = $results->result_id($result_index);
            my $result_len = $results->result_len($result_index);
            my $db_index = 0;
            while ($db_index < $result_len) {
                my %entry;
                $entry{'dbId'} = $dbId;
                $entry{'db'} = [
                    $results->result_get_db_start($result_index, $db_index),
                    $results->result_get_db_end($result_index, $db_index),
                ];
                $entry{'query'} = [
                    $results->result_get_query_start($result_index, $db_index),
                    $results->result_get_query_end($result_index, $db_index),
                ];
                push(@out, \%entry);

                $db_index++;
            }

            $result_index++;
        }
        return @out;
    } else {
        die("Error while running kmer_lookup: " . $result->get_err());
    }
});
$ffi->attach( DESTROY => ['KmerLookup'] => 'void');

package SW::KmerLookupBuilder;

our @ISA = qw( SW::KmerLookup );

$ffi->mangler(sub {
  my($name) = @_;
  "kmer_lookup_builder_$name";
});

$ffi->attach( new => ['string', 'u16'] => 'KmerLookupBuilder' );

$ffi->attach( kmer_step => ['KmerLookupBuilder', 'usize'] => 'i8' => sub {
    my ($xsub, $builder, $kmer_step) = @_;
    my $result = $xsub->($builder, $kmer_step);
    if ($result != 0) {
        die("invalid kmer step");
    }
});

$ffi->attach( max_sequence_distance => ['KmerLookupBuilder', 'f64'] => 'i8' => sub {
    my ($xsub, $builder, $distance) = @_;
    my $result = $xsub->($builder, $distance);
    if ($result != 0) {
        die("invalid max sequence distance");
    }
});

$ffi->attach( max_gc_diff => ['KmerLookupBuilder', 'f64']);
$ffi->attach( min_kmers => ['KmerLookupBuilder', 'usize']);
$ffi->attach( max_kmer_merge_distance => ['KmerLookupBuilder', 'usize']);
$ffi->attach( threads => ['KmerLookupBuilder', 'u16']);
$ffi->attach( max_reactivity => ['KmerLookupBuilder', 'f64']);
$ffi->attach( min_complexity => ['KmerLookupBuilder', 'f64']);
$ffi->attach( max_matches_every_nt => ['KmerLookupBuilder', 'usize']);
$ffi->attach( build => ['KmerLookupBuilder'] => 'KmerLookup');
$ffi->attach( DESTROY => ['KmerLookupBuilder'] => 'void');

package SW::KmerLookupOkErr;

our @ISA = qw( SW::KmerLookup );

$ffi->mangler(sub {
  my($name) = @_;
  "kmer_lookup_ok_err_$name";
});

$ffi->attach( is_ok => ['KmerLookupOkErr'] => 'u8' => sub {
    my($xsub, $kmer_lookup_result) = @_;
    return $xsub->($kmer_lookup_result) != 0;
});
$ffi->attach( get_ok => ['KmerLookupOkErr'] => 'KmerLookupResults' );
$ffi->attach( get_err => ['KmerLookupOkErr'] => 'string' );
$ffi->attach( DESTROY => ['KmerLookupOkErr'] => 'void');

package SW::KmerLookupResults;

our @ISA = qw( SW::KmerLookup );

$ffi->mangler(sub {
  my($name) = @_;
  "kmer_lookup_results_$name";
});

$ffi->attach( len => ['KmerLookupResults'] => 'usize' );
$ffi->attach( result_id => ['KmerLookupResults', 'usize'] => 'string' );
$ffi->attach( result_len => ['KmerLookupResults', 'usize'] => 'usize' );
$ffi->attach( result_get_db_start => ['KmerLookupResults', 'usize', 'usize'] => 'usize' );
$ffi->attach( result_get_db_end => ['KmerLookupResults', 'usize', 'usize'] => 'usize' );
$ffi->attach( result_get_query_start => ['KmerLookupResults', 'usize', 'usize'] => 'usize' );
$ffi->attach( result_get_query_end => ['KmerLookupResults', 'usize', 'usize'] => 'usize' );
$ffi->attach( DESTROY => ['KmerLookupResults'] => 'void' );
