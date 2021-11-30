package SW::Align;

use strict;
use SW::Align::Alignment;
use SW::Utils;
use Core::Mathematics qw(:all);
use Core::Utils;
use Data::Sequence::Utils;

use base qw(Core::Base);

sub new {

    my $class = shift;
    my %parameters = @_ if (@_);

    my $self = $class->SUPER::new(%parameters);
    $self->_init({ db              => [],
                   query           => [],
                   dbSeq           => undef,
                   querySeq        => undef,
                   match           => [-0.5, 2],
                   mismatch        => [-6, -0.5],
                   dbSeed          => [],
                   querySeed       => [],
                   gapOpen         => -14,
                   gapExt          => -5,
                   scoreSeq        => 0,
                   seqMatch        => 0.5,
                   seqMismatch     => -2,
                   winSize         => undef,
                   maxReactivity   => 1.0,
                   lenTollerance   => 0.1,
                   startScore      => 0,
                   maxDropOffRate  => 0.8,
                   maxDropOffBases => 8,
                   _winSize        => undef,
                   _db             => [],
                   _query          => [],
                   _dbSeq          => undef,
                   _querySeq       => undef }, \%parameters);

    $self->_validate();

    return($self);

}

sub _validate {

    my $self = shift;

    $self->throw("Invalid database seed") if (@{$self->{dbSeed}} != 2 || $self->{dbSeed}->[0] < 0 || $self->{dbSeed}->[1] > $#{$self->{db}});
    $self->throw("Invalid query seed") if (@{$self->{querySeed}} != 2 || $self->{querySeed}->[0] < 0 || $self->{querySeed}->[1] > $#{$self->{query}});
    $self->throw("Database and query seed must have the same length") if (abs(diff(@{$self->{dbSeed}})) != abs(diff(@{$self->{querySeed}})));

    if (!$self->{startScore}) {

        my (@db, @query);
        @db = map { $self->{db}->[$_] > $self->{maxReactivity} ? $self->{maxReactivity} : $self->{db}->[$_] } $self->{dbSeed}->[0] .. $self->{dbSeed}->[1];
        @query = map { $self->{query}->[$_] > $self->{maxReactivity} ? $self->{maxReactivity} : $self->{query}->[$_] } $self->{querySeed}->[0] .. $self->{querySeed}->[1];
        $self->{startScore} = sum(map { $self->_score($db[$_], $query[$_]) } 0 .. $#db);

        if ($self->{scoreSeq}) {

            my ($dbSeq, $querySeq);
            $dbSeq = substr($self->{dbSeq}, $self->{dbSeed}->[0], $self->{dbSeed}->[1] - $self->{dbSeed}->[0] + 1);
            $querySeq = substr($self->{querySeq}, $self->{querySeed}->[0], $self->{querySeed}->[1] - $self->{querySeed}->[0] + 1);
            $self->{startScore} += sum(map { substr($dbSeq, $_, 1) eq substr($querySeq, $_, 1) ? $self->{seqMatch} : $self->{seqMismatch} } 0 .. $#db);

        }

    }

}

sub align {

    my $self = shift;

    my ($regionLen, $extraLen, $dbRegionStart, $queryRegionStart,
        $dbRegionEnd, $queryRegionEnd, %partAln, %fullAln);

    return() if ($self->{startScore} <= 0);

    ####### Begin of upstream alignment #######

    $regionLen = min($self->{dbSeed}->[0], $self->{querySeed}->[0]);
    $extraLen = max(10, round($regionLen * $self->{lenTollerance})); # At least, we give 10 more bases to look into
    $regionLen += $extraLen;
    $self->{_winSize} = $self->{winSize} ? $self->{winSize} : $extraLen;
    $dbRegionStart = max(0, $self->{dbSeed}->[0] - $regionLen);
    $dbRegionEnd = $self->{dbSeed}->[0] - 1;
    $queryRegionStart = max(0, $self->{querySeed}->[0] - $regionLen);
    $queryRegionEnd = $self->{querySeed}->[0] - 1;
    $self->{_db} = [ reverse @{$self->{db}}[$dbRegionStart .. $dbRegionEnd] ];
    $self->{_query} = [ reverse @{$self->{query}}[$queryRegionStart .. $queryRegionEnd] ];
    $self->{_dbSeq} = reverse substr($self->{dbSeq}, $dbRegionStart, $dbRegionEnd - $dbRegionStart + 1);
    $self->{_querySeq} = reverse substr($self->{querySeq}, $queryRegionStart, $queryRegionEnd - $queryRegionStart + 1);
    %partAln = $self->_align($self->{startScore}, 1);

    ####### End of upstream alignment #######

    %fullAln = ( db         => [ @{$partAln{db}}, @{$self->{db}}[$self->{dbSeed}->[0] .. $self->{dbSeed}->[1]] ],
                 query      => [ @{$partAln{query}}, @{$self->{query}}[$self->{querySeed}->[0] .. $self->{querySeed}->[1]] ],
                 dbSeq      => $partAln{dbSeq} . substr($self->{dbSeq}, $self->{dbSeed}->[0], $self->{dbSeed}->[1] - $self->{dbSeed}->[0] + 1),
                 querySeq   => $partAln{querySeq} . substr($self->{querySeq}, $self->{querySeed}->[0], $self->{querySeed}->[1] - $self->{querySeed}->[0] + 1),
                 dbStart    => $partAln{score} ? $self->{dbSeed}->[0] - ($partAln{dbEnd} + 1) : $self->{dbSeed}->[0],
                 queryStart => $partAln{score} ? $self->{querySeed}->[0] - ($partAln{queryEnd} + 1) : $self->{querySeed}->[0],
                 score      => $partAln{score} ? $partAln{score} : $self->{startScore},
                 dbGaps     => [],
                 queryGaps  => [],
                 dbSeed     => $self->{dbSeed},
                 querySeed  => $self->{querySeed} );

    ####### Begin of downstream alignment #######

    $regionLen = min(scalar(@{$self->{db}}) - $self->{dbSeed}->[1], scalar(@{$self->{query}}) - $self->{querySeed}->[1]);
    $extraLen = max(10, round($regionLen * $self->{lenTollerance}));  # At least, we give 10 more bases to look into
    $regionLen += $extraLen;
    $dbRegionStart = $self->{dbSeed}->[1] + 1;
    $dbRegionEnd = min($self->{dbSeed}->[1] + $regionLen - 1, $#{$self->{db}});
    $queryRegionStart = $self->{querySeed}->[1] + 1;
    $queryRegionEnd = min($self->{querySeed}->[1] + $regionLen - 1, $#{$self->{query}});
    $self->{_winSize} = $self->{winSize} ? $self->{winSize} : $extraLen;
    $self->{_db} = [ @{$self->{db}}[$dbRegionStart .. $dbRegionEnd] ];
    $self->{_query} = [ @{$self->{query}}[$queryRegionStart .. $queryRegionEnd] ];
    $self->{_dbSeq} = substr($self->{dbSeq}, $dbRegionStart, $dbRegionEnd - $dbRegionStart + 1);
    $self->{_querySeq} = substr($self->{querySeq}, $queryRegionStart, $queryRegionEnd - $queryRegionStart + 1);
    %partAln = $self->_align($fullAln{score});

    ####### End of downstream alignment #######

    $fullAln{db} = [ @{$fullAln{db}}, @{$partAln{db}} ];
    $fullAln{query} = [ @{$fullAln{query}}, @{$partAln{query}} ];
    $fullAln{dbSeq} .= $partAln{dbSeq};
    $fullAln{querySeq} .= $partAln{querySeq};
    $fullAln{dbEnd} = $partAln{score} ? $self->{dbSeed}->[1] + $partAln{dbEnd} + 1 : $self->{dbSeed}->[1];
    $fullAln{queryEnd} = $partAln{score} ? $self->{querySeed}->[1] + $partAln{queryEnd} + 1 : $self->{querySeed}->[1];
    $fullAln{score} = $partAln{score} ? $partAln{score} : $self->{startScore};
    push(@{$fullAln{dbGaps}}, $-[0]) while($fullAln{dbSeq} =~ m/-/g);
    push(@{$fullAln{queryGaps}}, $-[0]) while($fullAln{querySeq} =~ m/-/g);

    return(SW::Align::Alignment->new(%fullAln));

}

sub _align {

    my $self = shift;
    my $best = shift;
    my $rev = shift if (@_);

    my ($gapOpen, $gapExt, $m, $n,
        @db, @query, @matrix, @best);
    $gapOpen = $self->{gapOpen};
    $gapExt = $self->{gapExt};
    @db = map { $_ > $self->{maxReactivity} ? $self->{maxReactivity} : $_ } @{$self->{_db}};
    @query = map { $_ > $self->{maxReactivity} ? $self->{maxReactivity} : $_ } @{$self->{_query}};
    $m = @db;
    $n = @query;

    # Init the matrix
    @matrix = map { [ map { { score => 0, ptr => "none", drop => 0 } } 0 .. $n ] } 0 .. $m;
    $matrix[0]->[0]->{score} = $best;
    $matrix[0]->[1]->{score} = max($best + $gapOpen + $gapExt, 0);
    $matrix[0]->[$_]->{score} = max($matrix[0]->[$_ - 1]->{score} + $gapExt, 0) for (2 .. $n);
    $matrix[1]->[0]->{score} = max($best + $gapOpen + $gapExt, 0);
    $matrix[$_]->[0]->{score} = max($matrix[$_ - 1]->[0]->{score} + $gapExt, 0) for (2 .. $m);

    # Populate the matrix
    for my $i (1 .. $m) {

        #my $bestSoFar = 0;

        for my $j (max(1, $i - $self->{_winSize}) .. min($i + $self->{_winSize}, $n)) {

            my ($score, $max, $ptr, $partialScore);
            $partialScore = $self->_score($db[$i - 1], $query[$j - 1]);
            $partialScore += substr($self->{_dbSeq}, $i - 1, 1) eq substr($self->{_querySeq}, $j - 1, 1) ? $self->{seqMatch} : $self->{seqMismatch} if ($self->{scoreSeq});
            $max = 0;
            $ptr = "none";

            if (($score = $matrix[$i - 1]->[$j - 1]->{score} + $partialScore) > $max) {

                $max = $score;
                $ptr = "diag";

            }

            if (($matrix[$i - 1]->[$j]->{ptr} eq "diag" &&
                 ($score = $matrix[$i - 1]->[$j]->{score} + $gapOpen + $gapExt) > $max) ||
                ($matrix[$i - 1]->[$j]->{ptr} eq "up" &&
                 ($score = $matrix[$i - 1]->[$j]->{score} + $gapExt) > $max)) {

                $max = $score;
                $ptr = "up";

            }

            if (($matrix[$i]->[$j - 1]->{ptr} eq "diag" &&
                 ($score = $matrix[$i]->[$j - 1]->{score} + $gapOpen + $gapExt) > $max) ||
                ($matrix[$i]->[$j - 1]->{ptr} eq "left" &&
                 ($score = $matrix[$i]->[$j - 1]->{score} + $gapExt) > $max)) {

                $max = $score;
                $ptr = "left";

            }

            # This ensures that, if the score drops to 0, no new path will start downstream of this
            # point, hence ensuring that the path will start from the origin
            next if (($ptr eq "diag" && !$matrix[$i - 1]->[$j - 1]->{score}) ||
                     ($ptr eq "left" && !$matrix[$i]->[$j - 1]->{score}) ||
                     ($ptr eq "up" && !$matrix[$i - 1]->[$j]->{score}));

            if ($max < $best * $self->{maxDropOffRate}) {

                if ($ptr eq "diag") { $matrix[$i]->[$j]->{drop} = $matrix[$i - 1]->[$j - 1]->{drop}; }
                elsif ($ptr eq "up") { $matrix[$i]->[$j]->{drop} = $matrix[$i - 1]->[$j]->{drop}; }
                elsif ($ptr eq "left") { $matrix[$i]->[$j]->{drop} = $matrix[$i]->[$j - 1]->{drop}; }

                $matrix[$i]->[$j]->{drop}++;

                if ($max < 0 || $matrix[$i]->[$j]->{drop} > $self->{maxDropOffBases}) {

                    $matrix[$i]->[$j]= { score => 0, ptr => "none", drop => 0 };

                    next;

                }

            }

            #$bestSoFar = $max if ($bestSoFar < $max);

            $matrix[$i]->[$j]->{score} = $max;
            $matrix[$i]->[$j]->{ptr} = $ptr;

            if ($max > $best) {

                $best = $max;
                @best = ($i, $j);

            }

        }

        #last if ($bestSoFar < $best * $self->{maxDropOffRate});

    }

    return($self->_traceback(\@best, \@matrix, $rev));

}

sub _debugMatrix { # call it to print out matrix (for debug)

    my $self = shift;
    my $matrix = shift;

    for(my $i = 0; $i <= $#{$matrix}; $i++) {

        for(my $j = 0; $j <= $#{$matrix->[$i]}; $j++) {

            print substr($matrix->[$i]->[$j]->{ptr}, 0, 1) . sprintf("%.2f", $matrix->[$i]->[$j]->{score}) . "_r" . $matrix->[$i]->[$j]->{drop} . "\t";

        }

        print "\n";

    }

    print "\n";

}

sub _score {

    my $self = shift;
    my ($db, $query) = @_;

    my $score = matchScore($db, $query, { match         => $self->{match},
                                          mismatch      => $self->{mismatch},
                                          maxReactivity => $self->{maxReactivity} });

    return($score);

}

sub _traceback {

    my $self = shift;
    my ($best, $matrix, $rev) = @_;

    my ($i, $j, $starti, $startj,
        $endi, $endj, $score, $db,
        $query, $dbSeq, $querySeq, @db,
        @query, %result);
    $db = $self->{_db};
    $query = $self->{_query};
    ($i, $j) = @{$best};
    ($endi, $endj) = ($i, $j);
    $score = $matrix->[$i]->[$j]->{score};

    if (@{$best}) {

        while($matrix->[$i]->[$j]->{ptr} ne "none") {

            ($starti, $startj) = ($i, $j);

            if ($matrix->[$i]->[$j]->{ptr} eq "diag") {

                $i--;
                $j--;

                my ($iBase, $jBase);
                $iBase = substr($self->{_dbSeq}, $i, 1);
                $jBase = substr($self->{_querySeq}, $j, 1);
                $dbSeq = $iBase . $dbSeq;
                $querySeq = $jBase . $querySeq;
                unshift(@db, $db->[$i]);
                unshift(@query, $query->[$j]);

            }
            elsif ($matrix->[$i]->[$j]->{ptr} eq "left") {

                $j--;

                $dbSeq = "-" . $dbSeq;
                $querySeq = substr($self->{_querySeq}, $j, 1) . $querySeq;
                unshift(@db, "NaN");
                unshift(@query, $query->[$j]);

            }
            elsif ($matrix->[$i]->[$j]->{ptr} eq "up") {

                $i--;

                $dbSeq = substr($self->{_dbSeq}, $i, 1) . $dbSeq;
                $querySeq = "-" . $querySeq;
                unshift(@db, $db->[$i]);
                unshift(@query, "NaN");

            }

        }

        if ($i > 0) {

            $score += $matrix->[0]->[0]->{score} + $self->{gapOpen} + $self->{gapExt} * $i if (!$matrix->[$i]->[$j]->{score});

            while($i > 0) {

                $dbSeq = substr($self->{_dbSeq}, $i - 1, 1) . $dbSeq;
                $querySeq = "-" . $querySeq;
                unshift(@db, $db->[$i - 1]);
                unshift(@query, "NaN");

                $i--;

            }

        }

        if ($j > 0) {

            $score += $matrix->[0]->[0]->{score} + $self->{gapOpen} + $self->{gapExt} * $j if (!$matrix->[$i]->[$j]->{score});

            while($j > 0) {

                $dbSeq = "-" . $dbSeq;
                $querySeq = substr($self->{_querySeq}, $j - 1, 1) . $querySeq;
                unshift(@db, "NaN");
                unshift(@query, $query->[$j - 1]);

                $j--;

            }

        }

        if ($rev) {

            $dbSeq = reverse $dbSeq;
            $querySeq = reverse $querySeq;
            @db = reverse @db;
            @query = reverse @query;

        }

    }
    else { $score = 0; }

    %result = $score > 0 ? ( db         => \@db,
                             query      => \@query,
                             dbSeq      => $dbSeq,
                             querySeq   => $querySeq,
                             dbEnd      => max(0, $endi - 1),
                             queryEnd   => max(0, $endj - 1),
                             score      => $score ) :
                           ( db         => [],
                             query      => [],
                             dbSeq      => undef,
                             querySeq   => undef,
                             dbEnd      => 0,
                             queryEnd   => 0,
                             score      => 0 );

    undef($self->{_db});
    undef($self->{_query});

    return(%result);

}

sub DESTROY {

    my $self = shift;

    undef($self->{db});
    undef($self->{query});
    undef($self->{_db});
    undef($self->{_query});

}

1;
