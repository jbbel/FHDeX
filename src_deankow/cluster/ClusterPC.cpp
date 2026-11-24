#include <algorithm>
#include <unordered_map>

#include <AMReX_ParallelDescriptor.H>

#include "ClusterPC.H"

using namespace amrex;

const Vector<std::string>&
ClusterPC::RealNames ()
{
    static const Vector<std::string> names {"mass", "sig2", "mu", "dm", "fx", "fy", "fz"};
    return names;
}

void
ClusterPC::SetFromState (CGState const& st)
{
    const int lev = 0;
    // drop whatever is there
    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        auto& aos = pti.GetArrayOfStructs();
        for (auto& p : aos()) { p.id() = -1; }
    }
    Redistribute();
    if (ParallelDescriptor::IOProcessor()) {
        // all into the first tile of the first local box; Redistribute sorts them out
        MFIter mfi = MakeMFIter(lev);
        AMREX_ALWAYS_ASSERT(mfi.isValid());
        auto& ptile = GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        for (int k = 0; k < st.size(); ++k) {
            ParticleType p;
            p.id() = st.ids[k];
            p.cpu() = 0;
            for (int d = 0; d < 3; ++d) { p.pos(d) = st.X[k][d]; }
            p.rdata(CGRealIdx::mass) = st.m[k];
            p.rdata(CGRealIdx::sig2) = st.s2[k];
            p.rdata(CGRealIdx::mu) = 0.0;
            p.rdata(CGRealIdx::dm) = 0.0;
            p.rdata(CGRealIdx::fx) = p.rdata(CGRealIdx::fy) = p.rdata(CGRealIdx::fz) = 0.0;
            ptile.push_back(p);
        }
    }
    Redistribute();
}

void
ClusterPC::Gather (CGState& st) const
{
    const int lev = 0;
    std::vector<Real> loc;
    for (ParConstIterType pti(*this, lev); pti.isValid(); ++pti) {
        const auto& aos = pti.GetArrayOfStructs();
        for (const auto& p : aos()) {
            if (p.id() < 0) { continue; }
            loc.push_back(static_cast<Real>(p.id()));
            for (int d = 0; d < 3; ++d) { loc.push_back(p.pos(d)); }
            loc.push_back(p.rdata(CGRealIdx::mass));
            loc.push_back(p.rdata(CGRealIdx::sig2));
        }
    }
    std::vector<Real> all;
#ifdef BL_USE_MPI
    const int nprocs = ParallelDescriptor::NProcs();
    int nloc = static_cast<int>(loc.size());
    std::vector<int> counts(nprocs), displs(nprocs);
    MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, ParallelDescriptor::Communicator());
    int total = 0;
    for (int r = 0; r < nprocs; ++r) { displs[r] = total; total += counts[r]; }
    all.resize(total);
    MPI_Allgatherv(loc.data(), nloc, ParallelDescriptor::Mpi_typemap<Real>::type(),
                   all.data(), counts.data(), displs.data(), ParallelDescriptor::Mpi_typemap<Real>::type(),
                   ParallelDescriptor::Communicator());
#else
    all = loc;
#endif
    const int K = static_cast<int>(all.size()/6);
    std::vector<int> order(K);
    for (int k = 0; k < K; ++k) { order[k] = k; }
    std::sort(order.begin(), order.end(), [&] (int a, int b) { return all[6*a] < all[6*b]; });
    st.ids.resize(K); st.X.resize(K); st.m.resize(K); st.s2.resize(K);
    for (int n = 0; n < K; ++n) {
        const Real* q = &all[6*order[n]];
        st.ids[n] = static_cast<long>(q[0]);
        st.X[n] = {q[1], q[2], q[3]};
        st.m[n] = q[4];
        st.s2[n] = q[5];
    }
}

void
ClusterPC::Sync (CGState const& st, const std::vector<Real>* mu, const std::vector<Real>* dm, const std::vector<Vec3>* f)
{
    const int lev = 0;
    std::unordered_map<long, int> idx;
    idx.reserve(2*st.size());
    for (int k = 0; k < st.size(); ++k) { idx[st.ids[k]] = k; }
    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        auto& aos = pti.GetArrayOfStructs();
        for (auto& p : aos()) {
            if (p.id() < 0) { continue; }
            auto it = idx.find(static_cast<long>(p.id()));
            if (it == idx.end()) { p.id() = -1; continue; }
            const int k = it->second;
            for (int d = 0; d < 3; ++d) { p.pos(d) = st.X[k][d]; }
            p.rdata(CGRealIdx::mass) = st.m[k];
            p.rdata(CGRealIdx::sig2) = st.s2[k];
            if (mu) { p.rdata(CGRealIdx::mu) = (*mu)[k]; }
            if (dm) { p.rdata(CGRealIdx::dm) = (*dm)[k]; }
            if (f) {
                p.rdata(CGRealIdx::fx) = (*f)[k][0];
                p.rdata(CGRealIdx::fy) = (*f)[k][1];
                p.rdata(CGRealIdx::fz) = (*f)[k][2];
            }
        }
    }
    Redistribute();
}
