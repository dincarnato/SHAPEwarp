use ExtUtils::MakeMaker;
use FFI::Build::MM;

my $fbmm = FFI::Build::MM->new;

WriteMakefile($fbmm->mm_args(
    ABSTRACT       => 'SHAPEwarp',
    DISTNAME       => 'SW-KmerLookup',
    NAME           => 'SW::KmerLookup',
    PARENTNAME     => 'SW::KmerLookup',
    VERSION_FROM   => 'lib/SW/KmerLookup.pm',
    BUILD_REQUIRES => {
        'FFI::Build::MM'          => '1.00',
        'FFI::Build::File::Cargo' => '0.07',
    },
    PREREQ_PM => {
        'FFI::Platypus'             => '1.00',
        'FFI::Platypus::Lang::Rust' => '0.07',
    },
    TEST_REQUIRES => {
        'Test2::V0' => '0',
    },
));

sub MY::postamble {
    $fbmm->mm_postamble;
}
