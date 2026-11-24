#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include "ClusterDriver.H"

using namespace amrex;

namespace {

constexpr Real TWO_PI = 2.0*M_PI;
const Real SQ2PI = std::sqrt(2.0/M_PI);

// truncated-Gaussian relations (dk_gaussfit.py): mass fraction and <r^2>/sigma^2 inside a standard deviations
Real trunc_mass_fraction (Real a) { return std::erf(a/std::sqrt(2.0)) - SQ2PI*a*std::exp(-0.5*a*a); }
Real trunc_r2_over_sigma2 (Real a)
{
    return (3.0*std::erf(a/std::sqrt(2.0)) - SQ2PI*(a*a*a + 3.0*a)*std::exp(-0.5*a*a))/trunc_mass_fraction(a);
}

std::vector<Vec3> lattice_bcc72 ()
{
    std::vector<Vec3> X;
    const Real a = 1.0/6;
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) for (int k = 0; k < 6; ++k) {
        X.push_back({i*a, j*a, k*a});
    }
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) for (int k = 0; k < 6; ++k) {
        X.push_back({i*a + a/2, j*a + a/2, k*a + a/2});
    }
    return X;
}

std::vector<Vec3> lattice_bcc7374 ()
{
    // direct lattice dual to the reciprocal basis G1 = (6,1,6), G2 = (6,6,-1), G3 = (0,7,5): A = G^{-T}
    const Real G[3][3] = {{6, 1, 6}, {6, 6, -1}, {0, 7, 5}};
    // inverse of G
    Real inv[3][3];
    const Real det = G[0][0]*(G[1][1]*G[2][2] - G[1][2]*G[2][1]) - G[0][1]*(G[1][0]*G[2][2] - G[1][2]*G[2][0])
                   + G[0][2]*(G[1][0]*G[2][1] - G[1][1]*G[2][0]);
    inv[0][0] =  (G[1][1]*G[2][2] - G[1][2]*G[2][1])/det;
    inv[0][1] = -(G[0][1]*G[2][2] - G[0][2]*G[2][1])/det;
    inv[0][2] =  (G[0][1]*G[1][2] - G[0][2]*G[1][1])/det;
    inv[1][0] = -(G[1][0]*G[2][2] - G[1][2]*G[2][0])/det;
    inv[1][1] =  (G[0][0]*G[2][2] - G[0][2]*G[2][0])/det;
    inv[1][2] = -(G[0][0]*G[1][2] - G[0][2]*G[1][0])/det;
    inv[2][0] =  (G[1][0]*G[2][1] - G[1][1]*G[2][0])/det;
    inv[2][1] = -(G[0][0]*G[2][1] - G[0][1]*G[2][0])/det;
    inv[2][2] =  (G[0][0]*G[1][1] - G[0][1]*G[1][0])/det;
    std::set<std::array<long, 3>> pts;
    for (int i = -14; i <= 14; ++i) for (int j = -14; j <= 14; ++j) for (int k = -14; k <= 14; ++k) {
        std::array<long, 3> key;
        for (int d = 0; d < 3; ++d) {
            Real x = inv[d][0]*i + inv[d][1]*j + inv[d][2]*k;
            x -= std::floor(x);
            long q = std::lround(x*1.e9) % 1000000000L;
            if (q < 0) { q += 1000000000L; }
            key[d] = q;
        }
        pts.insert(key);
    }
    std::vector<Vec3> X;
    for (auto const& k : pts) { X.push_back({k[0]*1.e-9, k[1]*1.e-9, k[2]*1.e-9}); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<long>(X.size()) == std::lround(std::abs(det)), "bcc7374: wrong site count");
    return X;
}

}

ClusterDriver::ClusterDriver ()
{
    ReadParameters();
}

ClusterDriver::~ClusterDriver ()
{
    for (std::FILE* fh : {fc, fs, fo, fe, fev, flog}) { if (fh) { std::fclose(fh); } }
}

void
ClusterDriver::Say (const std::string& s)
{
    amrex::Print() << s << "\n";
    if (flog) { std::fprintf(flog, "%s\n", s.c_str()); std::fflush(flog); }
}

void
ClusterDriver::ReadParameters ()
{
    read_potential_params(pot);
    {
        ParmParse pp;
        pp.get("diff_coeff", cp.D);
        pp.query("num_part", cp.N);
        int dorand = 0;
        pp.query("dorand", dorand);
        cp.pos_noise = (dorand != 0);
        int seed = 0;
        pp.query("seed", seed);
        cp.seed = static_cast<unsigned long>(seed);
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);
    }
    {
        ParmParse pp("cg");
        pp.query("init_file", init_file);
        pp.query("init_step", init_step);
        pp.query("init_t", init_t);
        pp.query("init_state", init_state);
        pp.query("lattice", lattice);
        pp.query("perturb_pos", perturb_pos);
        pp.query("perturb_mass", perturb_mass);
        pp.query("dt", dt_user);
        pp.query("out_dt", out_dt);
        pp.query("out_dir", out_dir);
        pp.query("fs", cp.fs);
        pp.query("gamma", cp.gamma);
        pp.query("b0", cp.b0);
        pp.query("rcut", cp.rcut);
        pp.query("r_nb", cp.r_nb);
        pp.query("gabriel_beta", cp.gabriel_beta);
        pp.query("nseg", cp.nseg);
        pp.query("newton", cp.newton);
        pp.query("saddle_int", cp.saddle_int);
        pp.query("rebuild_int", cp.rebuild_int);
        pp.query("barrier_min", cp.barrier_min);
        pp.query("dm_cap", cp.dm_cap);
        pp.query("dt_kappa", cp.dt_kappa);
        pp.query("width_iters", cp.width_iters);
        pp.query("m_min", cp.m_min);
        pp.query("merge_frac", cp.merge_frac);
        pp.query("merge_rmax", cp.merge_rmax);
        pp.query("wide_frac", cp.wide_frac);
        int mn = 0, pn = -1;
        pp.query("mass_noise", mn);
        pp.query("pos_noise", pn);
        cp.mass_noise = (mn != 0);
        if (pn >= 0) { cp.pos_noise = (pn != 0); }
        int spill = 1;
        pp.query("spill", spill);
        cp.spill = (spill != 0);
        pp.query("table_file", table_file);
        pp.query("write_state", write_state);
        pp.query("write_particles", write_particles);
        pp.query("render", render);
    }
    Vector<int> n_cell(3, 128);
    Vector<int> max_grid_size(1, 128);
    {
        ParmParse pp("amr");
        pp.queryarr("n_cell", n_cell);
        pp.queryarr("max_grid_size", max_grid_size);
        pp.query("plot_file", plot_file);
        pp.query("plot_int", plot_int);
        pp.query("plot_dt", plot_dt);
        pp.query("check_file", check_file);
        pp.query("check_int", check_int);
        pp.query("restart", restart_dir);
    }
    Box domain(IntVect(0, 0, 0), IntVect(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));
    RealBox rb({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    Array<int, 3> is_per {1, 1, 1};
    geom.define(domain, rb, CoordSys::cartesian, is_per);
    grids.define(domain);
    grids.maxSize(max_grid_size[0]);
    dmap.define(grids);
}

void
ClusterDriver::InitData ()
{
    if (ParallelDescriptor::IOProcessor()) {
        mkdir(out_dir.c_str(), 0755);
        flog = std::fopen((out_dir + "/run.log").c_str(), "a");
    }
    wall0 = amrex::second();
    // the smoothed-potential table
    if (!table_file.empty() && table.read(table_file)) {
        Say("W table read from " + table_file);
    } else {
        const double t0 = amrex::second();
        table.build(pot);
        std::ostringstream os;
        os << "W table built in " << amrex::second() - t0 << " s: " << table.nr() << " x " << table.ns()
           << ", Uhat(0) = " << table.uhat0;
        Say(os.str());
        if (!table_file.empty() && ParallelDescriptor::IOProcessor()) { table.write(table_file); }
    }
    model = std::make_unique<CGModel>(table, cp);
    pc = std::make_unique<ClusterPC>(geom, dmap, grids);

    if (!restart_dir.empty()) {
        Restart();
    } else {
        InitState();
        pc->SetFromState(st);
    }
    {
        std::ostringstream os;
        os << "model: D " << cp.D << " N " << cp.N << " f_s(b) = " << cp.fs << " exp(-" << cp.gamma << " (b - " << cp.b0
           << ")) dt " << dt_user << " m_min " << cp.m_min << " merge " << cp.merge_frac << "/" << cp.merge_rmax
           << " barrier_min " << cp.barrier_min << " noise pos " << cp.pos_noise << " mass " << cp.mass_noise;
        Say(os.str());
    }
    Rebuild();
    model->widths(st);
    {
        Real smean = 0.0, s2m = 0.0;
        for (Real v : st.s2) { smean += std::sqrt(v); s2m += v; }
        smean /= st.size();
        std::ostringstream os;
        os.precision(5);
        os << "sigma " << smean << "; stiffness estimate " << kappa << ", dt*kappa = " << dt_user*kappa
           << " (explicit Euler needs < 2); " << pairs.ngab << " Gabriel pairs, " << pairs.i.size() << " pairs within rcut";
        Say(os.str());
    }
    OpenTables();
}

void
ClusterDriver::InitState ()
{
    st = CGState();
    if (!init_state.empty()) {
        std::ifstream in(init_state);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(in.good(), "cannot open cg.init_state");
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) { continue; }
            std::istringstream is(line);
            if (line[0] == '#') { std::string hash; is >> hash >> st.t >> st.step; continue; }
            long id; Vec3 x; Real m, s2;
            is >> id >> x[0] >> x[1] >> x[2] >> m >> s2;
            st.ids.push_back(id); st.X.push_back(x); st.m.push_back(m); st.s2.push_back(s2);
        }
        std::ostringstream os;
        os << "init from state " << init_state << ": " << st.size() << " clusters at t " << st.t;
        Say(os.str());
    } else if (!init_file.empty()) {
        // DK clusters.txt: t step id mass xc yc zc Rg peak ...; the frame at init_step, or nearest to init_t
        std::ifstream in(init_file);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(in.good(), "cannot open cg.init_file");
        struct Row { Real t; long step; long id; Real m, x, y, z, Rg, peak; };
        std::vector<Row> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') { continue; }
            std::istringstream is(line);
            Row r;
            if (is >> r.t >> r.step >> r.id >> r.m >> r.x >> r.y >> r.z >> r.Rg >> r.peak) { rows.push_back(r); }
        }
        long step = init_step;
        if (step < 0) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(init_t >= 0.0, "give cg.init_step or cg.init_t");
            Real best = std::numeric_limits<Real>::max();
            for (auto const& r : rows) { if (std::abs(r.t - init_t) < best) { best = std::abs(r.t - init_t); step = r.step; } }
        }
        Real msum = 0.0;
        for (auto const& r : rows) {
            if (r.step != step) { continue; }
            const Real a = std::sqrt(2.0*std::log(std::max(r.peak/5.0, 1.0 + 1.e-12)));
            const Real fr = trunc_mass_fraction(a);
            const Real g = trunc_r2_over_sigma2(a);
            st.ids.push_back(r.id);
            st.X.push_back({r.x, r.y, r.z});
            st.m.push_back(r.m/fr);
            st.s2.push_back(r.Rg*r.Rg/g);
            st.t = r.t;
            msum += r.m/fr;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(st.size() > 0, "no rows at cg.init_step in cg.init_file");
        for (Real& v : st.m) { v /= msum; }
        st.step = 0;
        std::ostringstream os;
        os.precision(5);
        os << "init from " << init_file << " at step " << step << " (t " << st.t << "): " << st.size()
           << " clusters, tail-corrected mass sum " << msum << " renormalised to 1";
        Say(os.str());
    } else {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!lattice.empty(), "give cg.init_file, cg.init_state or cg.lattice");
        std::vector<Vec3> X = (lattice == "bcc72") ? lattice_bcc72() : lattice_bcc7374();
        std::mt19937_64 rng(cp.seed);
        std::normal_distribution<Real> nrm(0.0, 1.0);
        const int K = static_cast<int>(X.size());
        for (int k = 0; k < K; ++k) {
            if (perturb_pos > 0) { for (int d = 0; d < 3; ++d) { X[k][d] += perturb_pos*nrm(rng); X[k][d] -= std::floor(X[k][d]); } }
            st.ids.push_back(k + 1);
            st.X.push_back(X[k]);
            st.m.push_back(1.0/K);
            st.s2.push_back(1.e-4);
        }
        if (perturb_mass > 0) {
            Real s = 0.0;
            for (Real& v : st.m) { v *= 1.0 + perturb_mass*nrm(rng); s += v; }
            for (Real& v : st.m) { v /= s; }
        }
        std::ostringstream os;
        os << "init " << lattice << ": " << K << " sites";
        Say(os.str());
    }
}

// ------------------------------------------------------------------------------------------------
// the prototype's Simulator
// ------------------------------------------------------------------------------------------------
void
ClusterDriver::Rebuild ()
{
    // warm start for the Gabriel pairs that survive, matched by the ids of the two members
    std::vector<Vec3> warm;
    const bool had = sad_valid;
    std::vector<std::pair<long, long>> old_keys;
    std::vector<Vec3> old_xs;
    if (had) {
        old_keys.reserve(sad.i.size());
        for (std::size_t n = 0; n < sad.i.size(); ++n) {
            // ids of the previous indexing were stored in sad at the time (see Prepare): recover them via the
            // saved id arrays
            old_keys.emplace_back(sad_ids_i[n], sad_ids_j[n]);
        }
        old_xs = sad.xs;
    }
    model->build_pairs(st, pairs);
    pairs_valid = true;
    kappa = model->kappa_X(st, pairs);
    sad_valid = false;
    if (had) {
        std::map<std::pair<long, long>, int> pos;
        for (std::size_t n = 0; n < old_keys.size(); ++n) { pos[old_keys[n]] = static_cast<int>(n); }
        warm.reserve(pairs.ngab);
        bool all = true;
        for (std::size_t q = 0; q < pairs.i.size() && all; ++q) {
            if (!pairs.gab[q]) { continue; }
            auto it = pos.find({st.ids[pairs.i[q]], st.ids[pairs.j[q]]});
            if (it == pos.end()) { all = false; break; }
            warm.push_back(old_xs[it->second]);
        }
        if (all) {
            model->saddles(st, pairs, sad, &warm);
            sad_ids_i.resize(sad.i.size()); sad_ids_j.resize(sad.i.size());
            for (std::size_t n = 0; n < sad.i.size(); ++n) { sad_ids_i[n] = st.ids[sad.i[n]]; sad_ids_j[n] = st.ids[sad.j[n]]; }
            sad_valid = true;
        }
    }
}

void
ClusterDriver::Prepare ()
{
    if (!pairs_valid) { Rebuild(); }
    model->widths(st);
    model->mu(st, pairs, mu);
    model->forces(st, f);
    if (!sad_valid || st.step % cp.saddle_int == 0) {
        const std::vector<Vec3>* warm = (sad_valid && static_cast<int>(sad.xs.size()) == pairs.ngab) ? &sad.xs : nullptr;
        std::vector<Vec3> wcopy;
        if (warm) { wcopy = *warm; warm = &wcopy; }
        model->saddles(st, pairs, sad, warm);
        sad_ids_i.resize(sad.i.size()); sad_ids_j.resize(sad.i.size());
        for (std::size_t n = 0; n < sad.i.size(); ++n) { sad_ids_i[n] = st.ids[sad.i[n]]; sad_ids_j[n] = st.ids[sad.j[n]]; }
        sad_valid = true;
    }
    have_mu = true;
}

void
ClusterDriver::Step (Real dt)
{
    Prepare();
    if (kappa > 0.0) { dt = std::min(dt, cp.dt_kappa/kappa); }
    model->exchange(st, mu, sad, dt, dm, Jij, Jji);
    const int K = st.size();
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < 3; ++d) {
            Real dx = f[k][d]/st.m[k]*dt;
            if (cp.pos_noise) { dx += model->normal()*std::sqrt(2.0*cp.D*dt/(cp.N*st.m[k])); }
            st.X[k][d] += dx;
            st.X[k][d] -= std::floor(st.X[k][d]);
        }
    }
    for (int k = 0; k < K; ++k) {
        st.m[k] += dm[k];
        if (st.m[k] <= 0.0) {
            std::ostringstream os;
            os << "non-positive mass at step " << st.step << " t " << st.t << " for id " << st.ids[k] << " (m " << st.m[k] << ")";
            Say(os.str());
            amrex::Abort(os.str());
        }
    }
    st.t += dt;
    st.step += 1;
    const bool changed = model->handle_events(st, pairs, sad, mu, Jij, Jji, events, [this] (const std::string& s) { Say(s); });
    if (changed || st.step % cp.rebuild_int == 0) { Rebuild(); }
}

void
ClusterDriver::Evolve ()
{
    Real t_next = st.t;
    Real next_plot = (plot_dt > 0.0) ? st.t + plot_dt : std::numeric_limits<Real>::max();
    long nsteps = 0;
    while (st.t < stop_time - 1.e-12 && nsteps < max_step) {
        if (st.t >= t_next - 1.e-12) {
            if (!have_mu || !sad_valid) { Prepare(); }
            if (dm.size() != mu.size()) { dm.assign(mu.size(), 0.0); }
            WriteFrame();
            t_next += out_dt;
        }
        Step(std::min(dt_user, stop_time - st.t));
        ++nsteps;
        const bool plot_now = (plot_int > 0 && st.step % plot_int == 0) || (plot_dt > 0.0 && st.t >= next_plot*(1.0 - 1.e-12));
        if (plot_now) {
            if (plot_dt > 0.0 && st.t >= next_plot*(1.0 - 1.e-12)) { next_plot += plot_dt; }
            pc->Sync(st, &mu, &dm, &f);
            WritePlotFile();
        }
        if (check_int > 0 && st.step % check_int == 0) {
            pc->Sync(st, &mu, &dm, &f);
            Checkpoint();
        }
    }
    Prepare();
    if (dm.size() != mu.size()) { dm.assign(mu.size(), 0.0); }
    WriteFrame();
    pc->Sync(st, &mu, &dm, &f);
    if (plot_int > 0 || plot_dt > 0.0) { WritePlotFile(); }
    if (check_int > 0) { Checkpoint(); }
    std::ostringstream os;
    os.precision(4);
    os << std::fixed << "run done: t = " << st.t << ", " << st.step << " steps, " << st.size() << " clusters, "
       << events.size() << " events, " << std::setprecision(0) << amrex::second() - wall0 << " s";
    Say(os.str());
}

// ------------------------------------------------------------------------------------------------
// output
// ------------------------------------------------------------------------------------------------
void
ClusterDriver::OpenTables ()
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    const bool append = !restart_dir.empty();
    const char* mode = append ? "a" : "w";
    fc = std::fopen((out_dir + "/clusters.txt").c_str(), mode);
    fs = std::fopen((out_dir + "/summary.txt").c_str(), mode);
    fo = std::fopen((out_dir + "/order.txt").c_str(), mode);
    fe = std::fopen((out_dir + "/fe_cg.txt").c_str(), mode);
    fev = std::fopen((out_dir + "/events.txt").c_str(), mode);
    if (!append) {
        std::fprintf(fc, "# t step id mass xc yc zc Rg peak ncells wide q4 q6\n");
        std::fprintf(fs, "# t step nclusters mass_fraction largest_mass merges splits forms dissolves\n");
        std::fprintf(fo, "# t step nclusters q4 q6 Q4_global Q6_global kpeak nmodes PR top_share\n");
        std::fprintf(fe, "# t step F-Funi F_ent F_self F_pair mass_sum m_max m_min sigma_mean mu_spread kappa_max\n");
        std::fprintf(fev, "# t step type old_ids old_masses new_ids new_masses\n");
    }
}

void
ClusterDriver::WriteFrame ()
{
    const int K = st.size();
    Real f_ent, f_self, f_pair;
    const Real F = model->free_energy(st, f_ent, f_self, f_pair);
    Real msum = 0.0, mmax = 0.0, mmin = std::numeric_limits<Real>::max(), smean = 0.0, mum = 0.0, mu2 = 0.0;
    for (int k = 0; k < K; ++k) {
        msum += st.m[k]; mmax = std::max(mmax, st.m[k]); mmin = std::min(mmin, st.m[k]);
        smean += std::sqrt(st.s2[k]); mum += mu[k]; mu2 += mu[k]*mu[k];
    }
    smean /= K; mum /= K;
    const Real mu_spread = std::sqrt(std::max(mu2/K - mum*mum, 0.0));
    int nok = 0;
    Real bsum = 0.0;
    std::vector<Real> barriers;
    for (std::size_t n = 0; n < sad.i.size(); ++n) {
        if (sad.ok[n]) { ++nok; barriers.push_back((sad.Us[n] - mu[sad.i[n]])/cp.D); }
    }
    Real bmed = 0.0;
    if (!barriers.empty()) { std::nth_element(barriers.begin(), barriers.begin() + barriers.size()/2, barriers.end()); bmed = barriers[barriers.size()/2]; }
    Real frms = 0.0, mmean = msum/K;
    for (int k = 0; k < K; ++k) { frms += f[k][0]*f[k][0] + f[k][1]*f[k][1] + f[k][2]*f[k][2]; }
    frms = std::sqrt(frms/K)/mmean;
    (void) bsum;
    {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "t %.4f step %ld K %d  F-Funi %.5f  sigma %.5f  mu spread %.4f  saddles ok %d/%zu barrier %.2f D  |f/m| rms %.3f  m min/max %.2e %.2e  kappa %.0f capped %d spill %d  events %zu  %.0f s",
                      st.t, st.step, K, F - 0.5*table.uhat0, smean, mu_spread, nok, sad.i.size(), bmed, frms, mmin, mmax, kappa,
                      model->n_capped(), model->n_spill(), events.size(), amrex::second() - wall0);
        Say(buf);
    }
    if (!ParallelDescriptor::IOProcessor()) { return; }
    int nm = 0, nd = 0;
    for (std::size_t e = events_written; e < events.size(); ++e) {
        auto const& ev = events[e];
        if (ev.type == "merge") { ++nm; } else { ++nd; }
        std::fprintf(fev, "%.10g %ld %s ", ev.t, ev.step, ev.type.c_str());
        for (std::size_t q = 0; q < ev.old_ids.size(); ++q) { std::fprintf(fev, "%s%ld", q ? "," : "", ev.old_ids[q]); }
        std::fprintf(fev, " ");
        for (std::size_t q = 0; q < ev.old_m.size(); ++q) { std::fprintf(fev, "%s%.6g", q ? "," : "", ev.old_m[q]); }
        std::fprintf(fev, " ");
        for (std::size_t q = 0; q < ev.new_ids.size(); ++q) { std::fprintf(fev, "%s%ld", q ? "," : "", ev.new_ids[q]); }
        std::fprintf(fev, " ");
        for (std::size_t q = 0; q < ev.new_m.size(); ++q) { std::fprintf(fev, "%s%.6g", q ? "," : "", ev.new_m[q]); }
        std::fprintf(fev, "\n");
    }
    events_written = events.size();
    for (int k = 0; k < K; ++k) {
        // DK-equivalent threshold quantities of a Gaussian: peak, cells above 5 x mean, Rg of the truncated Gaussian
        const Real sig = std::sqrt(st.s2[k]);
        const Real peak = st.m[k]/std::pow(TWO_PI*st.s2[k], 1.5);
        const Real a = std::sqrt(std::max(2.0*std::log(std::max(peak/5.0, 1.0 + 1.e-9)), 1.e-9));
        const Real rthr = sig*a;
        const long ncells = static_cast<long>(4.0*M_PI/3.0*rthr*rthr*rthr*128.0*128.0*128.0);
        const Real Rg = sig*std::sqrt(trunc_r2_over_sigma2(a));
        std::fprintf(fc, "%.10g %ld %ld %.6g %.8g %.8g %.8g %.6g %.6g %ld 0 0 0\n",
                     st.t, st.step, st.ids[k], st.m[k], st.X[k][0], st.X[k][1], st.X[k][2], Rg, peak, ncells);
    }
    std::fprintf(fs, "%.10g %ld %d %.6f %.6g %d 0 0 %d\n", st.t, st.step, K, msum, mmax, nm, nd);
    std::fprintf(fo, "%.10g %ld %d 0 0 0 0 0 0 0 0\n", st.t, st.step, K);
    std::fprintf(fe, "%.10g %ld %.8g %.8g %.8g %.8g %.12g %.6g %.6g %.6g %.6g %.6g\n",
                 st.t, st.step, F - 0.5*table.uhat0, f_ent, f_self, f_pair, msum, mmax, mmin, smean, mu_spread, kappa);
    for (std::FILE* fh : {fc, fs, fo, fe, fev}) { std::fflush(fh); }
    if (write_state) { WriteState(); }
}

void
ClusterDriver::WriteState ()
{
    char name[64];
    std::snprintf(name, sizeof(name), "/state_%06ld.txt", st.step);
    std::FILE* fh = std::fopen((out_dir + name).c_str(), "w");
    if (!fh) { return; }
    std::fprintf(fh, "# %.17g %ld\n", st.t, st.step);
    for (int k = 0; k < st.size(); ++k) {
        std::fprintf(fh, "%ld %.17g %.17g %.17g %.17g %.17g\n", st.ids[k], st.X[k][0], st.X[k][1], st.X[k][2], st.m[k], st.s2[k]);
    }
    std::fclose(fh);
}

void
ClusterDriver::RenderDensity (MultiFab& phi) const
{
    // cell averages of the Gaussian density sum m_k G(x - X_k), normalised to mean 1 like the DK phi
    const auto dx = geom.CellSizeArray();
    const Box& domain = geom.Domain();
    const IntVect n = domain.size();
    phi.setVal(0.0);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& a = phi.array(mfi);
        for (int k = 0; k < st.size(); ++k) {
            const Real sig = std::sqrt(st.s2[k]);
            const Real half = 4.5*sig;
            const Real inv = 1.0/(std::sqrt(2.0)*sig);
            int lo[3], hi[3];
            for (int d = 0; d < 3; ++d) {
                lo[d] = static_cast<int>(std::floor((st.X[k][d] - half)/dx[d]));
                hi[d] = static_cast<int>(std::floor((st.X[k][d] + half)/dx[d]));
            }
            for (int i = lo[0]; i <= hi[0]; ++i) {
                const Real wx = 0.5*(std::erf(((i+1)*dx[0] - st.X[k][0])*inv) - std::erf((i*dx[0] - st.X[k][0])*inv));
                const int iw = ((i % n[0]) + n[0]) % n[0];
                for (int j = lo[1]; j <= hi[1]; ++j) {
                    const Real wy = 0.5*(std::erf(((j+1)*dx[1] - st.X[k][1])*inv) - std::erf((j*dx[1] - st.X[k][1])*inv));
                    const int jw = ((j % n[1]) + n[1]) % n[1];
                    for (int l = lo[2]; l <= hi[2]; ++l) {
                        const int lw = ((l % n[2]) + n[2]) % n[2];
                        if (!bx.contains(IntVect(iw, jw, lw))) { continue; }
                        const Real wz = 0.5*(std::erf(((l+1)*dx[2] - st.X[k][2])*inv) - std::erf((l*dx[2] - st.X[k][2])*inv));
                        a(iw, jw, lw) += st.m[k]*wx*wy*wz/(dx[0]*dx[1]*dx[2]);
                    }
                }
            }
        }
    }
}

void
ClusterDriver::WritePlotFile ()
{
    const std::string name = amrex::Concatenate(plot_file, static_cast<int>(st.step), 6);
    Say("Writing plotfile " + name);
    if (render) {
        MultiFab phi(grids, dmap, 1, 0);
        RenderDensity(phi);
        WriteSingleLevelPlotfile(name, phi, {"phi0"}, geom, st.t, static_cast<int>(st.step));
    } else {
        MultiFab dummy(grids, dmap, 1, 0);
        dummy.setVal(0.0);
        WriteSingleLevelPlotfile(name, dummy, {"phi0"}, geom, st.t, static_cast<int>(st.step));
    }
    if (write_particles) {
        pc->WritePlotFile(name, "clusters", ClusterPC::RealNames(), Vector<std::string>());
    }
}

void
ClusterDriver::Checkpoint ()
{
    const std::string name = amrex::Concatenate(check_file, static_cast<int>(st.step), 6);
    Say("Writing checkpoint " + name);
    if (ParallelDescriptor::IOProcessor()) {
        amrex::UtilCreateDirectory(name, 0755);
        std::ofstream h(name + "/Header");
        h.precision(17);
        h << st.t << " " << st.step << " " << events.size() << "\n";
    }
    ParallelDescriptor::Barrier();
    pc->Checkpoint(name, "clusters");
}

void
ClusterDriver::Restart ()
{
    Say("Restarting from " + restart_dir);
    std::ifstream h(restart_dir + "/Header");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(h.good(), "cannot read checkpoint Header");
    std::size_t nev;
    h >> st.t >> st.step >> nev;
    pc->Restart(restart_dir, "clusters");
    pc->Gather(st);
    std::ostringstream os;
    os << "restart: " << st.size() << " clusters at t " << st.t << " step " << st.step;
    Say(os.str());
}
