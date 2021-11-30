package SW::Align::Alignment;

use strict;
use Core::Mathematics qw(:all);
use Core::Utils;
use Data::Sequence::Utils;

use base qw(Core::Base);

sub new {

    my $class = shift;
    my %parameters = @_ if (@_);

    my $self = $class->SUPER::new(%parameters);
    $self->_init({ dbId            => undef,
                   queryId         => undef,
                   db              => [],
                   query           => [],
                   dbSeq           => undef,
                   querySeq        => undef,
                   dbStart         => undef,
                   dbEnd           => undef,
                   queryStart      => undef,
                   queryEnd        => undef,
                   dbGaps          => [],
                   queryGaps       => [],
                   dbSeed          => [],
                   querySeed       => [],
                   score           => undef,
                   evalue          => undef,
                   pvalue          => undef,
                   alignFoldPvalue => undef,
                   structure       => undef,
                   sci             => undef,
		   dbBpSupport     => undef,
		   queryBpSupport  => undef }, \%parameters);

    $self->_validate();

    return($self);

}

sub _validate {

    my $self = shift;

}

sub db {

    my $self = shift;

    return(map {sprintf("%.2f", $_) } @{$self->{db}});

}

sub query {

    my $self = shift;

    return(map {sprintf("%.2f", $_) } @{$self->{query}});

}

sub dbSeq {

    my $self = shift;

    return($self->{dbSeq});

}

sub querySeq {

    my $self = shift;

    return($self->{querySeq});

}

sub dbId {

    my $self = shift;
    my $id = shift if (@_);

    $self->{dbId} = $id if (defined $id);

    return($self->{dbId});

}

sub queryId {

    my $self = shift;
    my $id = shift if (@_);

    $self->{queryId} = $id if (defined $id);

    return($self->{queryId});

}

sub score {

    my $self = shift;
    my $score = shift if (@_);

    $self->{score} = $score if (defined $score && isnumeric($score));

    return($self->{score});

}

sub evalue {

    my $self = shift;
    my $evalue = shift if (@_);

    $self->{evalue} = $evalue if (defined $evalue && ispositive($evalue));

    return($self->{evalue});

}

sub pvalue {

    my $self = shift;
    my $pvalue = shift if (@_);

    $self->{pvalue} = $pvalue if (defined $pvalue && inrange($pvalue, [0, 1]));

    return($self->{pvalue});

}

sub alignFoldPvalue {

    my $self = shift;
    my $pvalue = shift if (@_);

    $self->{alignFoldPvalue} = $pvalue if (defined $pvalue && inrange($pvalue, [0, 1]));

    return($self->{alignFoldPvalue});

}

sub structure {

    my $self = shift;
    my $structure = shift if (@_);

    $self->{structure} = $structure if (defined $structure);

    return($self->{structure});

}

sub sci {

    my $self = shift;
    my $sci = shift if (@_);

    $self->{sci} = $sci if (defined $sci && ispositive($sci));

    return($self->{sci});

}

sub dbBpSupport {

    my $self = shift;
    my $dbBpSupport = shift if (@_);

    $self->{dbBpSupport} = $dbBpSupport if (defined $dbBpSupport && ispositive($dbBpSupport));

    return($self->{dbBpSupport});

}

sub queryBpSupport {

    my $self = shift;
    my $queryBpSupport = shift if (@_);

    $self->{queryBpSupport} = $queryBpSupport if (defined $queryBpSupport && ispositive($queryBpSupport));

    return($self->{queryBpSupport});

}

sub dbStart {

    my $self = shift;
    my $start = shift if (@_);

    $self->{dbStart} = $start if (defined $start && isnumeric($start));

    return($self->{dbStart});

}

sub dbEnd {

    my $self = shift;
    my $end = shift if (@_);

    $self->{dbEnd} = $end if (defined $end && isnumeric($end));

    return($self->{dbEnd});

}

sub queryStart {

    my $self = shift;
    my $start = shift if (@_);

    $self->{queryStart} = $start if (defined $start && isnumeric($start));

    return($self->{queryStart});

}

sub queryEnd {

    my $self = shift;
    my $end = shift if (@_);

    $self->{queryEnd} = $end if (defined $end && isnumeric($end));

    return($self->{queryEnd});

}

sub dbSeed { return(wantarray() ? @{$_[0]->{dbSeed}} : $_[0]->{dbSeed}); }

sub querySeed { return(wantarray() ? @{$_[0]->{querySeed}} : $_[0]->{querySeed}); }

sub dbGaps { return(wantarray() ? @{$_[0]->{dbGaps}} : $_[0]->{dbGaps}); }

sub queryGaps { return(wantarray() ? @{$_[0]->{queryGaps}} : $_[0]->{queryGaps}); }

sub dbNonGaps {

    my $self = shift;

    my (%gaps, @nonGaps);
    %gaps = map { $_ => 1 } @{$self->{dbGaps}};
    @nonGaps = grep { !exists $gaps{$_} } 0 .. $#{$self->{db}};

    return(wantarray() ? @nonGaps : \@nonGaps);

}

sub queryNonGaps {

    my $self = shift;

    my (%gaps, @nonGaps);
    %gaps = map { $_ => 1 } @{$self->{queryGaps}};
    @nonGaps = grep { !exists $gaps{$_} } 0 .. $#{$self->{query}};

    return(wantarray() ? @nonGaps : \@nonGaps);

}

sub DESTROY {

    my $self = shift;

    delete($self->{db});
    delete($self->{query});
    delete($self->{alignment});

}

1;
