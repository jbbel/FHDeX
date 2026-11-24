#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>

#include "ClusterDriver.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main()");
        const auto strt_total = amrex::second();

        ClusterDriver driver;
        driver.InitData();
        driver.Evolve();

        auto end_total = amrex::second() - strt_total;
        ParallelDescriptor::ReduceRealMax(end_total, ParallelDescriptor::IOProcessorNumber());
        amrex::Print() << "\nTotal Time: " << end_total << '\n';
    }
    amrex::Finalize();
}
