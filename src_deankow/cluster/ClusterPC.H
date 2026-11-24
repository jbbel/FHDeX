#ifndef CLUSTER_PC_H_
#define CLUSTER_PC_H_

#include <vector>

#include <AMReX_Particles.H>

#include "CGModel.H"

// Real attributes of a cluster particle; id() holds the DK cluster id
struct CGRealIdx
{
    enum {
        mass = 0,
        sig2,
        mu,
        dm,
        fx, fy, fz,
        ncomps
    };
};

// The clusters as AMReX particles: storage, plotfiles, checkpoints and redistribution. The
// dynamics is computed on the gathered global state (CGState, identical on every rank: a few
// hundred clusters), and written back here by id (Sync). Gather rebuilds the global state from
// the particles (at the start and after a restart).
class ClusterPC
    : public amrex::ParticleContainer<CGRealIdx::ncomps, 0>
{
public:

    using Base = amrex::ParticleContainer<CGRealIdx::ncomps, 0>;

    ClusterPC (const amrex::Geometry& geom, const amrex::DistributionMapping& dmap, const amrex::BoxArray& ba)
        : Base(geom, dmap, ba) {}

    // create the particles of a global state (on the I/O rank, then redistributed)
    void SetFromState (CGState const& st);

    // the global state from the particles (all ranks get the same, sorted by id)
    void Gather (CGState& st) const;

    // positions, masses, widths (and the diagnostics mu, dm, f when given) from the state; particles
    // whose id is no longer in the state are removed; then Redistribute
    void Sync (CGState const& st, const std::vector<amrex::Real>* mu = nullptr,
               const std::vector<amrex::Real>* dm = nullptr, const std::vector<Vec3>* f = nullptr);

    static const amrex::Vector<std::string>& RealNames ();
};

#endif
