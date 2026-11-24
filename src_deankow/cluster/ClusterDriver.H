#ifndef CLUSTER_DRIVER_H_
#define CLUSTER_DRIVER_H_

#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "CGModel.H"
#include "CGTable.H"
#include "ClusterPC.H"
#include "Potentials.H"

// Driver of the cluster-level model: parameters, initialisation (DK cluster table, text state,
// or a reference lattice), the time loop of the prototype (integrate.py), the DK-format tables
// (clusters, summary, order, events, fe_cg), the rendered density plotfiles and checkpoints.
class ClusterDriver
{
public:

    ClusterDriver ();
    ~ClusterDriver ();

    void InitData ();
    void Evolve ();

private:

    void ReadParameters ();
    void InitState ();
    void Restart ();
    void Checkpoint ();

    // the prototype's Simulator
    void Rebuild ();
    void Prepare ();
    void Step (amrex::Real dt);

    // output
    void OpenTables ();
    void WriteFrame ();
    void WriteState ();
    void WritePlotFile ();
    void RenderDensity (amrex::MultiFab& phi) const;
    void Say (const std::string& s);

    PotentialParams pot;
    CGParams cp;
    CGTable table;
    std::unique_ptr<CGModel> model;

    CGState st;
    CGPairs pairs;
    bool pairs_valid = false;
    CGSaddles sad;
    std::vector<long> sad_ids_i, sad_ids_j;   // ids of the saddle pairs (for warm starts across rebuilds)
    bool sad_valid = false;
    amrex::Real kappa = 0.0;
    std::vector<amrex::Real> mu, dm, Jij, Jji;
    std::vector<Vec3> f;
    bool have_mu = false;
    std::vector<CGEvent> events;
    std::size_t events_written = 0;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;
    std::unique_ptr<ClusterPC> pc;

    // parameters
    std::string init_file, init_state, lattice, restart_dir, table_file;
    long init_step = -1;
    amrex::Real init_t = -1.0;
    amrex::Real perturb_pos = 0.0, perturb_mass = 0.0;
    amrex::Real dt_user = 5.e-5;
    amrex::Real out_dt = 0.005;
    amrex::Real stop_time = 1.0;
    long max_step = std::numeric_limits<long>::max();
    std::string out_dir = ".";
    std::string plot_file = "plt_cg_";
    std::string check_file = "chk_cg_";
    int plot_int = -1;
    amrex::Real plot_dt = -1.0;
    int check_int = -1;
    int write_state = 1;
    int write_particles = 1;
    int render = 1;

    std::FILE* fc = nullptr; std::FILE* fs = nullptr; std::FILE* fo = nullptr;
    std::FILE* fe = nullptr; std::FILE* fev = nullptr; std::FILE* flog = nullptr;
    double wall0 = 0.0;
};

#endif
