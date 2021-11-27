use Test2::V0;
use File::Basename;
use File::Spec;
use SW;

my $sw = KmerLookup->new(
    kmerLen => 12,
    maxKmerDist => 10,
    maxMatchesEveryNt => 200,
    maxReactivity => 1.5,
    minComplexity => 0.2,
    minKmers => 2,
);
my @query = (
    0.052, 0.046, 0.108, 0.241, 0.221, 1.224, 0.246, 0.846, 1.505, 0.627, 0.078, 0.002, 0.056,
    0.317, 0.114, 0.157, 0.264, 1.016, 2.925, 2.205, 1.075, 1.210, 0.191, 0.016, 0.045, 0.015,
    0.087, 0.572, 0.052, 0.157, 0.796, 2.724, 0.027, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004,
    0.003, 0.063, 0.144, 0.072, 0.054, 0.096, 0.112, 0.002, 0.000, 0.019, 0.026, 0.021, 1.022,
    2.108, 0.111, 0.000, 0.007, 0.000, 0.002, 0.000, 0.010, 0.037, 0.078, 0.152, 0.355, 1.738,
    0.715, 0.211, 0.179, 0.036, 0.046, 0.159, 0.257, 0.312, 0.931, 0.798, 0.618, 0.935, 0.147,
    0.015, 0.014, 0.031, 0.147, 0.149, 0.577, 1.052, 1.410, 0.487, 0.636, 0.238, 0.286, 0.462,
    1.586, 1.683, 0.597, 1.165, 1.265, 2.094, 0.422, 0.462, 1.900, 4.055, 0.481, 0.511, 0.087,
    1.217, 1.180, 0.094, 0.018, 0.033, 0.081, 0.148, 0.163, 0.160, 1.019, 0.339, 0.507, 1.039,
    0.824, 0.122, 0.420, 0.429, 0.913, 1.383, 0.610, 0.417, 0.825, 0.743, 0.433, 0.401, 0.993,
    0.497, 0.404, 0.407, 0.316, 0.017, 0.005, 0.046, 0.072, 0.037, 0.091, 0.282, 0.203, 0.033,
    0.004, 0.021, 0.262, 0.157, 0.050, 0.019, 0.059, 0.102, 0.083, 0.066, 0.040, 0.075, 0.061,
    0.573, 0.631, 0.427, 0.265, 1.190, 0.066, 0.042, 0.085, 0.424, 0.413, 0.375, 0.447, 0.035,
    0.045, 0.037, 0.242, 0.221, 0.157, 0.170, 0.370, 1.238, 0.743, 0.571, 0.138, 0.837, 0.859,
    0.042, 0.021, 0.080, 0.318, 0.195, 0.792, 1.581, 1.058, 2.004, 1.512, 2.273, 1.256, 0.036,
    0.005, 0.094, 0.091, 0.464, 0.741,
);
my $query_sequence = "GATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATCTGATACTGGCAAGCTTGAGTCTCGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGGCCCCCTGGACGAAGACTGACGCTCAGGTGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGG";

my $result = $sw->run(File::Spec->catfile(dirname(__FILE__), "test.db"), \@query, $query_sequence);
is $result, 286;

done_testing;
