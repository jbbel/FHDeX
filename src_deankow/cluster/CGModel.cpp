#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "CGModel.H"

using namespace amrex;

namespace {
constexpr Real TWO_PI = 2.0*M_PI;
}

// ------------------------------------------------------------------------------------------------
// geometry
// ------------------------------------------------------------------------------------------------
Vec3
CGModel::min_image (Vec3 const& d)
{
    Vec3 o;
    for (int a = 0; a < 3; ++a) { o[a] = d[a] - std::floor(d[a] + 0.5); }
    return o;
}

Real
CGModel::norm (Vec3 const& d)
{
    return std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
}

void
CGModel::build_pairs (CGState const& st, CGPairs& pr) const
{
    const int K = st.size();
    pr.i.clear(); pr.j.clear(); pr.d.clear(); pr.r.clear(); pr.gab.clear();
    pr.adj.assign(K, {});
    const Real rc2 = m_p.rcut*m_p.rcut;
    for (int a = 0; a < K; ++a) {
        for (int b = a + 1; b < K; ++b) {
            Vec3 d = min_image({st.X[b][0] - st.X[a][0], st.X[b][1] - st.X[a][1], st.X[b][2] - st.X[a][2]});
            const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            if (r2 < rc2) {
                pr.i.push_back(a); pr.j.push_back(b); pr.d.push_back(d); pr.r.push_back(std::sqrt(r2));
                pr.adj[a].push_back(b); pr.adj[b].push_back(a);
            }
        }
    }
    // Gabriel test on the candidates within r_nb: no third centre within beta r/2 of the midpoint;
    // a blocking centre lies within r of both members, so the adjacency lists suffice
    const int P = static_cast<int>(pr.i.size());
    pr.gab.assign(P, 0);
    pr.ngab = 0;
    for (int q = 0; q < P; ++q) {
        if (pr.r[q] >= m_p.r_nb) { continue; }
        const int a = pr.i[q], b = pr.j[q];
        Vec3 mid;
        for (int c = 0; c < 3; ++c) { mid[c] = st.X[a][c] + 0.5*pr.d[q][c]; }
        const Real lim = 0.5*m_p.gabriel_beta*pr.r[q];
        bool blocked = false;
        for (int pass = 0; pass < 2 && !blocked; ++pass) {
            for (int c : pr.adj[pass == 0 ? a : b]) {
                if (c == a || c == b) { continue; }
                Vec3 dm = min_image({st.X[c][0] - mid[0], st.X[c][1] - mid[1], st.X[c][2] - mid[2]});
                if (norm(dm) <= lim) { blocked = true; break; }
            }
        }
        if (!blocked) { pr.gab[q] = 1; ++pr.ngab; }
    }
}

// ------------------------------------------------------------------------------------------------
// statics
// ------------------------------------------------------------------------------------------------
void
CGModel::widths (CGState& st) const
{
    // cage rule: s2_j = 3 D / Lap Phi(X_j), Phi smoothed by the contributors' s2 only, self term
    // included; iterated width_iters times from the current widths, clamped to the table range
    const int K = st.size();
    const Real lo = m_tab.s2min()*0.51, hi = m_tab.s2max()*0.49;
    const Real rc2 = m_p.rcut*m_p.rcut;
    std::vector<Real> lap(K);
    for (int it = 0; it < m_p.width_iters; ++it) {
        for (int a = 0; a < K; ++a) { lap[a] = st.m[a]*m_tab.lap(0.0, st.s2[a]); }
        for (int a = 0; a < K; ++a) {
            for (int b = a + 1; b < K; ++b) {
                Vec3 d = min_image({st.X[b][0] - st.X[a][0], st.X[b][1] - st.X[a][1], st.X[b][2] - st.X[a][2]});
                const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
                if (r2 >= rc2) { continue; }
                const Real r = std::sqrt(r2);
                lap[a] += st.m[b]*m_tab.lap(r, st.s2[b]);
                lap[b] += st.m[a]*m_tab.lap(r, st.s2[a]);
            }
        }
        for (int a = 0; a < K; ++a) {
            st.s2[a] = std::min(std::max(3.0*m_p.D/lap[a], lo), hi);
        }
    }
}

void
CGModel::mu (CGState const& st, CGPairs const& pr, std::vector<Real>& mu) const
{
    const int K = st.size();
    mu.resize(K);
    std::vector<Real> phi(K);
    for (int a = 0; a < K; ++a) { phi[a] = st.m[a]*m_tab.W(0.0, st.s2[a]); }
    for (std::size_t q = 0; q < pr.i.size(); ++q) {
        const int a = pr.i[q], b = pr.j[q];
        Vec3 d = min_image({st.X[b][0] - st.X[a][0], st.X[b][1] - st.X[a][1], st.X[b][2] - st.X[a][2]});
        const Real r = norm(d);
        phi[a] += st.m[b]*m_tab.W(r, st.s2[b]);
        phi[b] += st.m[a]*m_tab.W(r, st.s2[a]);
    }
    for (int a = 0; a < K; ++a) {
        mu[a] = m_p.D*(std::log(st.m[a]) - 1.5*std::log(TWO_PI*st.s2[a])) - 0.5*m_p.D + phi[a];
    }
}

void
CGModel::forces (CGState const& st, std::vector<Vec3>& f) const
{
    const int K = st.size();
    f.assign(K, {0.0, 0.0, 0.0});
    const Real rc2 = m_p.rcut*m_p.rcut;
    for (int a = 0; a < K; ++a) {
        for (int b = a + 1; b < K; ++b) {
            Vec3 d = min_image({st.X[b][0] - st.X[a][0], st.X[b][1] - st.X[a][1], st.X[b][2] - st.X[a][2]});
            const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            if (r2 >= rc2) { continue; }
            const Real r = std::sqrt(r2);
            Real W, Wr, Ws;
            m_tab.eval(r, st.s2[a] + st.s2[b], W, Wr, Ws);
            const Real c = st.m[a]*st.m[b]*Wr/std::max(r, 1.e-300);
            for (int k = 0; k < 3; ++k) { f[a][k] += c*d[k]; f[b][k] -= c*d[k]; }
        }
    }
}

Real
CGModel::free_energy (CGState const& st, Real& f_ent, Real& f_self, Real& f_pair) const
{
    const int K = st.size();
    f_ent = f_self = f_pair = 0.0;
    for (int a = 0; a < K; ++a) {
        f_ent += st.m[a]*(std::log(st.m[a]) - 1.5*std::log(TWO_PI*M_E*st.s2[a]));
        f_self += 0.5*st.m[a]*st.m[a]*m_tab.W(0.0, 2.0*st.s2[a]);
    }
    f_ent *= m_p.D;
    const Real rc2 = m_p.rcut*m_p.rcut;
    for (int a = 0; a < K; ++a) {
        for (int b = a + 1; b < K; ++b) {
            Vec3 d = min_image({st.X[b][0] - st.X[a][0], st.X[b][1] - st.X[a][1], st.X[b][2] - st.X[a][2]});
            const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            if (r2 >= rc2) { continue; }
            f_pair += st.m[a]*st.m[b]*m_tab.W(std::sqrt(r2), st.s2[a] + st.s2[b]);
        }
    }
    return f_ent + f_self + f_pair;
}

Real
CGModel::kappa_X (CGState const& st, CGPairs const& pr) const
{
    std::vector<Real> k(st.size(), 0.0);
    for (std::size_t q = 0; q < pr.i.size(); ++q) {
        const int a = pr.i[q], b = pr.j[q];
        const Real r = std::max(pr.r[q], 1.e-12);
        Real W, Wr, Ws;
        m_tab.eval(pr.r[q], st.s2[a] + st.s2[b], W, Wr, Ws);
        const Real Wrr = 2.0*Ws - 2.0*Wr/r;
        const Real c = std::max(std::abs(Wrr), std::abs(Wr/r));
        k[a] += st.m[b]*c;
        k[b] += st.m[a]*c;
    }
    Real kmax = 0.0;
    for (Real v : k) { kmax = std::max(kmax, v); }
    return kmax;
}

// ------------------------------------------------------------------------------------------------
// field, saddles, exchange
// ------------------------------------------------------------------------------------------------
void
CGModel::field (Vec3 const& x, CGState const& st, std::vector<int> const& contrib, int order,
                Real& phi, Vec3& g, std::array<Real, 9>& H) const
{
    phi = 0.0;
    g = {0.0, 0.0, 0.0};
    H.fill(0.0);
    Real c2 = 0.0;
    for (int c : contrib) {
        Vec3 d = min_image({x[0] - st.X[c][0], x[1] - st.X[c][1], x[2] - st.X[c][2]});   // from contributor to point
        const Real r = norm(d);
        const Real rs = std::max(r, 1.e-12);
        Real W, Wr, Ws;
        m_tab.eval(r, st.s2[c], W, Wr, Ws);
        const Real mc = st.m[c];
        phi += mc*W;
        if (order < 1) { continue; }
        const Real a = mc*Wr/rs;
        for (int k = 0; k < 3; ++k) { g[k] += a*d[k]; }
        if (order < 2) { continue; }
        const Real Wrr = 2.0*Ws - 2.0*Wr/rs;
        const Real c1 = mc*(Wrr - Wr/rs)/(rs*rs);
        c2 += a;
        for (int A = 0; A < 3; ++A) {
            for (int B = A; B < 3; ++B) {
                H[3*A + B] += c1*d[A]*d[B];
            }
        }
    }
    if (order >= 2) {
        for (int A = 0; A < 3; ++A) {
            H[3*A + A] += c2;
            for (int B = A + 1; B < 3; ++B) { H[3*B + A] = H[3*A + B]; }
        }
    }
}

void
eig3_sym (std::array<Real, 9> const& Hin, Vec3& lam)
{
    // cyclic Jacobi rotations
    std::array<Real, 9> A = Hin;
    for (int sweep = 0; sweep < 50; ++sweep) {
        Real off = std::abs(A[1]) + std::abs(A[2]) + std::abs(A[5]);
        Real diag = std::abs(A[0]) + std::abs(A[4]) + std::abs(A[8]);
        if (off <= 1.e-300 || off < 1.e-16*diag) { break; }
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                const Real apq = A[3*p + q];
                if (apq == 0.0) { continue; }
                const Real theta = (A[3*q + q] - A[3*p + p])/(2.0*apq);
                const Real t = (theta >= 0 ? 1.0 : -1.0)/(std::abs(theta) + std::sqrt(theta*theta + 1.0));
                const Real c = 1.0/std::sqrt(t*t + 1.0), s = t*c;
                for (int k = 0; k < 3; ++k) {
                    const Real akp = A[3*k + p], akq = A[3*k + q];
                    A[3*k + p] = c*akp - s*akq;
                    A[3*k + q] = s*akp + c*akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const Real apk = A[3*p + k], aqk = A[3*q + k];
                    A[3*p + k] = c*apk - s*aqk;
                    A[3*q + k] = s*apk + c*aqk;
                }
            }
        }
    }
    lam = {A[0], A[4], A[8]};
    std::sort(lam.begin(), lam.end());
}

Vec3
solve3 (std::array<Real, 9> H, Vec3 const& b, Real eps)
{
    // Gaussian elimination with partial pivoting on (H + eps I) x = b
    Real M[3][4];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) { M[r][c] = H[3*r + c] + (r == c ? eps : 0.0); }
        M[r][3] = b[r];
    }
    for (int c = 0; c < 3; ++c) {
        int piv = c;
        for (int r = c + 1; r < 3; ++r) { if (std::abs(M[r][c]) > std::abs(M[piv][c])) { piv = r; } }
        if (piv != c) { for (int k = 0; k < 4; ++k) { std::swap(M[c][k], M[piv][k]); } }
        const Real p = M[c][c];
        if (p == 0.0) { return {0.0, 0.0, 0.0}; }
        for (int r = c + 1; r < 3; ++r) {
            const Real f = M[r][c]/p;
            for (int k = c; k < 4; ++k) { M[r][k] -= f*M[c][k]; }
        }
    }
    Vec3 x;
    for (int r = 2; r >= 0; --r) {
        Real s = M[r][3];
        for (int k = r + 1; k < 3; ++k) { s -= M[r][k]*x[k]; }
        x[r] = s/M[r][r];
    }
    return x;
}

void
CGModel::saddles (CGState const& st, CGPairs const& pr, CGSaddles& sad, const std::vector<Vec3>* warm) const
{
    const int G = pr.ngab;
    sad.sel.clear(); sad.i.clear(); sad.j.clear();
    sad.sel.reserve(G); sad.i.reserve(G); sad.j.reserve(G);
    for (std::size_t q = 0; q < pr.i.size(); ++q) {
        if (pr.gab[q]) { sad.sel.push_back(static_cast<int>(q)); sad.i.push_back(pr.i[q]); sad.j.push_back(pr.j[q]); }
    }
    sad.xs.assign(G, {0.0, 0.0, 0.0});
    sad.Us.assign(G, 0.0);
    sad.A.assign(G, std::numeric_limits<Real>::quiet_NaN());
    sad.gnorm.assign(G, 0.0);
    sad.Useg.assign(G, 0.0);
    sad.lam.assign(G, {0.0, 0.0, 0.0});
    sad.ok.assign(G, 0);
    const bool use_warm = (warm != nullptr && static_cast<int>(warm->size()) == G);
    const int newton = use_warm ? std::max(2, m_p.newton/3) : m_p.newton;
    const Real rmax2 = 0.22*0.22;

    std::vector<int> contrib;
    std::array<Real, 9> H;
    Vec3 g;
    Real phi;
    for (int n = 0; n < G; ++n) {
        const int a = sad.i[n], b = sad.j[n];
        // current displacement and midpoint
        Vec3 d = min_image({st.X[b][0] - st.X[a][0], st.X[b][1] - st.X[a][1], st.X[b][2] - st.X[a][2]});
        const Real r = norm(d);
        Vec3 mid;
        for (int k = 0; k < 3; ++k) { mid[k] = st.X[a][k] + 0.5*d[k]; }
        // contributors: the two members and their neighbour lists, within 0.22 of the midpoint, unique
        contrib.clear();
        contrib.push_back(a); contrib.push_back(b);
        for (int c : pr.adj[a]) { contrib.push_back(c); }
        for (int c : pr.adj[b]) { contrib.push_back(c); }
        std::sort(contrib.begin(), contrib.end());
        contrib.erase(std::unique(contrib.begin(), contrib.end()), contrib.end());
        contrib.erase(std::remove_if(contrib.begin(), contrib.end(), [&] (int c) {
            Vec3 dm = min_image({st.X[c][0] - mid[0], st.X[c][1] - mid[1], st.X[c][2] - mid[2]});
            return dm[0]*dm[0] + dm[1]*dm[1] + dm[2]*dm[2] > rmax2;
        }), contrib.end());

        // highest sample of Phi on the segment between 0.25 and 0.75: the ridge height along the straight
        // path (kept for the spill test) and, without a warm start, the initial guess for the saddle
        Vec3 xs, xbest = {0.0, 0.0, 0.0};
        Real best = -std::numeric_limits<Real>::infinity();
        for (int s = 0; s < m_p.nseg; ++s) {
            const Real tt = 0.25 + 0.5*s/(m_p.nseg - 1);
            Vec3 p;
            for (int k = 0; k < 3; ++k) { p[k] = st.X[a][k] + tt*d[k]; }
            field(p, st, contrib, 0, phi, g, H);
            if (phi > best) { best = phi; xbest = p; }
        }
        sad.Useg[n] = best;
        xs = use_warm ? (*warm)[n] : xbest;
        for (int it = 0; it < newton; ++it) {
            field(xs, st, contrib, 2, phi, g, H);
            Vec3 step = solve3(H, g, 1.e-9);
            Real nrm = norm(step);
            const Real sc = std::min(1.0, 0.01/std::max(nrm, 1.e-300));
            for (int k = 0; k < 3; ++k) { xs[k] = xs[k] - sc*step[k]; xs[k] -= std::floor(xs[k]); }
        }
        field(xs, st, contrib, 2, phi, g, H);
        Vec3 lam;
        eig3_sym(H, lam);
        const Real gn = norm(g);
        Vec3 off = min_image({xs[0] - mid[0], xs[1] - mid[1], xs[2] - mid[2]});
        const bool ok = (lam[0] < 0) && (lam[1] > 0) && (norm(off) < 0.6*r) && (gn < 1.e-3*std::abs(lam[0])*r);
        sad.xs[n] = xs;
        sad.Us[n] = phi;
        sad.lam[n] = lam;
        sad.gnorm[n] = gn;
        sad.ok[n] = ok;
        if (ok) { sad.A[n] = m_p.D*std::sqrt(TWO_PI*m_p.D)*std::sqrt(-lam[0]/(lam[1]*lam[2])); }
    }
}

void
CGModel::exchange (CGState const& st, std::vector<Real> const& mu, CGSaddles const& sad, Real dt,
                   std::vector<Real>& dm, std::vector<Real>& Jij, std::vector<Real>& Jji)
{
    const int K = st.size();
    const int G = static_cast<int>(sad.i.size());
    dm.assign(K, 0.0);
    Jij.assign(G, 0.0);
    Jji.assign(G, 0.0);
    m_ncapped = 0;
    m_nspill = 0;
    const Real gexp = 1.0 + m_p.gamma;
    // reference prefactor for the spill channels: the median A of the accepted saddles
    Real A0 = 0.0;
    {
        std::vector<Real> As;
        for (int n = 0; n < G; ++n) { if (sad.ok[n]) { As.push_back(sad.A[n]); } }
        if (!As.empty()) { std::nth_element(As.begin(), As.begin() + As.size()/2, As.end()); A0 = As[As.size()/2]; }
    }
    const Real Jspill = m_p.fs*A0*std::exp(-gexp*m_p.barrier_min + m_p.gamma*m_p.b0);   // flux of a channel at the barrier floor
    for (int n = 0; n < G; ++n) {
        const int a = sad.i[n], b = sad.j[n];
        const bool act = sad.ok[n] && (sad.Us[n] - std::max(mu[a], mu[b]) > m_p.barrier_min*m_p.D);
        if (!act) {
            // no accepted barrier between the pair: a cluster whose mu reaches the ridge of Phi along the
            // straight path to its neighbour (within the barrier floor) has no barrier to cross and spills mass
            // down the gradient at the flux of a channel at the floor, instead of being frozen by the
            // activation rule (merged clusters far above their neighbours). Ordinary pairs without an accepted
            // saddle keep a ridge several D above both members and stay closed.
            const Real ridge = sad.Useg[n];
            if (m_p.spill && mu[a] > mu[b] && ridge - mu[a] < m_p.barrier_min*m_p.D) { Jij[n] = Jspill; ++m_nspill; }
            else if (m_p.spill && mu[b] > mu[a] && ridge - mu[b] < m_p.barrier_min*m_p.D) { Jji[n] = Jspill; ++m_nspill; }
            else { continue; }
        } else {
            const Real bi = (sad.Us[n] - mu[a])/m_p.D, bj = (sad.Us[n] - mu[b])/m_p.D;
            Jij[n] = m_p.fs*sad.A[n]*std::exp(-gexp*bi + m_p.gamma*m_p.b0);
            Jji[n] = m_p.fs*sad.A[n]*std::exp(-gexp*bj + m_p.gamma*m_p.b0);
        }
        Real delta = (Jji[n] - Jij[n])*dt;                           // mass gained by i from j
        const Real cap = m_p.dm_cap*std::min(st.m[a], st.m[b]);
        if (std::abs(delta) > cap) { ++m_ncapped; delta = (delta > 0 ? cap : -cap); }
        if (m_p.mass_noise) {
            delta += normal()*std::sqrt(std::max(Jij[n] + Jji[n], 0.0)*dt/m_p.N);
        }
        dm[a] += delta;
        dm[b] -= delta;
    }
}

// ------------------------------------------------------------------------------------------------
// events
// ------------------------------------------------------------------------------------------------
bool
CGModel::handle_events (CGState& st, CGPairs const& pr, CGSaddles const& sad, std::vector<Real> const& mu,
                        std::vector<Real> const& Jij, std::vector<Real> const& Jji,
                        std::vector<CGEvent>& events, std::function<void(const std::string&)> const& log)
{
    const int K = st.size();
    const int P = static_cast<int>(pr.i.size());
    std::vector<char> remove(K, 0);
    bool changed = false;
    char buf[512];

    // current pair distances
    std::vector<Real> r(P);
    std::vector<Vec3> d(P);
    for (int q = 0; q < P; ++q) {
        const int a = pr.i[q], b = pr.j[q];
        d[q] = min_image({st.X[b][0] - st.X[a][0], st.X[b][1] - st.X[a][1], st.X[b][2] - st.X[a][2]});
        r[q] = norm(d[q]);
    }
    // barrier of each pair (Gabriel pairs with a valid saddle; -inf for Gabriel pairs without one)
    std::vector<Real> barrier(P, std::numeric_limits<Real>::infinity());
    for (std::size_t n = 0; n < sad.sel.size(); ++n) {
        barrier[sad.sel[n]] = sad.ok[n] ? sad.Us[n] - std::max(mu[sad.i[n]], mu[sad.j[n]])
                                        : -std::numeric_limits<Real>::infinity();
    }
    // mergers: centres closer than merge_frac (sigma_i + sigma_j), or closer than merge_rmax with no barrier
    std::vector<int> close;
    for (int q = 0; q < P; ++q) {
        const int a = pr.i[q], b = pr.j[q];
        const bool c1 = r[q] < m_p.merge_frac*(std::sqrt(st.s2[a]) + std::sqrt(st.s2[b]));
        const bool c2 = (r[q] < m_p.merge_rmax) && (barrier[q] < m_p.barrier_min*m_p.D);
        if (c1 || c2) { close.push_back(q); }
    }
    std::stable_sort(close.begin(), close.end(), [&] (int p, int q) { return r[p] < r[q]; });
    for (int q : close) {
        const int a = pr.i[q], b = pr.j[q];
        if (remove[a] || remove[b]) { continue; }
        const int keep = (st.ids[a] < st.ids[b]) ? a : b;
        const int gone = (keep == a) ? b : a;
        const Real mt = st.m[a] + st.m[b];
        Vec3 dd = min_image({st.X[gone][0] - st.X[keep][0], st.X[gone][1] - st.X[keep][1], st.X[gone][2] - st.X[keep][2]});
        Vec3 Xn;
        for (int k = 0; k < 3; ++k) { Xn[k] = st.X[keep][k] + st.m[gone]/mt*dd[k]; Xn[k] -= std::floor(Xn[k]); }
        const Real dd2 = dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2];
        const Real s2n = (st.m[a]*st.s2[a] + st.m[b]*st.s2[b])/mt + st.m[a]*st.m[b]*dd2/(3.0*mt*mt);
        const bool by_width = r[q] < m_p.merge_frac*(std::sqrt(st.s2[a]) + std::sqrt(st.s2[b]));
        CGEvent ev;
        ev.t = st.t; ev.step = st.step; ev.type = "merge";
        ev.old_ids = {st.ids[a], st.ids[b]}; ev.old_m = {st.m[a], st.m[b]};
        ev.new_ids = {st.ids[keep]}; ev.new_m = {mt};
        events.push_back(ev);
        std::snprintf(buf, sizeof(buf), "merge ids %ld + %ld at t %.5f: r %.4f, masses %.3e %.3e%s",
                      st.ids[a], st.ids[b], st.t, r[q], st.m[a], st.m[b], by_width ? "" : " [no barrier]");
        log(buf);
        st.X[keep] = Xn; st.m[keep] = mt; st.s2[keep] = s2n;
        st.m[gone] = 0.0;
        remove[gone] = 1;
        changed = true;
    }
    // dissolution: m < m_min, or width larger than a fraction of the nearest distance
    std::vector<Real> rmin(K, std::numeric_limits<Real>::infinity());
    for (int q = 0; q < P; ++q) {
        rmin[pr.i[q]] = std::min(rmin[pr.i[q]], r[q]);
        rmin[pr.j[q]] = std::min(rmin[pr.j[q]], r[q]);
    }
    for (int a = 0; a < K; ++a) {
        if (remove[a]) { continue; }
        const bool wide = std::sqrt(st.s2[a]) > m_p.wide_frac*rmin[a];
        if (!(st.m[a] < m_p.m_min || wide)) { continue; }
        std::snprintf(buf, sizeof(buf), "dissolve id %ld at t %.5f: m %.3e (m_min %.1e), sigma %.4f, nearest centre %.4f%s",
                      st.ids[a], st.t, st.m[a], m_p.m_min, std::sqrt(st.s2[a]), rmin[a], wide ? " [width criterion]" : "");
        log(buf);
        // the mass goes to the Gabriel partners in proportion to the outgoing flux
        std::vector<int> partners;
        std::vector<Real> wts;
        for (std::size_t n = 0; n < sad.i.size(); ++n) {
            if (!sad.ok[n]) { continue; }
            if (sad.i[n] == a && !remove[sad.j[n]]) { partners.push_back(sad.j[n]); wts.push_back(Jij[n]); }
        }
        for (std::size_t n = 0; n < sad.i.size(); ++n) {
            if (!sad.ok[n]) { continue; }
            if (sad.j[n] == a && !remove[sad.i[n]]) { partners.push_back(sad.i[n]); wts.push_back(Jji[n]); }
        }
        if (partners.empty()) { continue; }
        Real wsum = std::accumulate(wts.begin(), wts.end(), 0.0);
        if (wsum <= 0.0) { std::fill(wts.begin(), wts.end(), 1.0); wsum = static_cast<Real>(wts.size()); }
        CGEvent ev;
        ev.t = st.t; ev.step = st.step; ev.type = "dissolve";
        ev.old_ids = {st.ids[a]}; ev.old_m = {st.m[a]};
        for (std::size_t n = 0; n < partners.size(); ++n) {
            st.m[partners[n]] += st.m[a]*wts[n]/wsum;
            ev.new_ids.push_back(st.ids[partners[n]]);
            ev.new_m.push_back(st.m[partners[n]]);
        }
        events.push_back(ev);
        st.m[a] = 0.0;
        remove[a] = 1;
        changed = true;
    }
    if (changed) {
        CGState nw;
        nw.t = st.t; nw.step = st.step;
        for (int a = 0; a < K; ++a) {
            if (remove[a]) { continue; }
            nw.ids.push_back(st.ids[a]); nw.X.push_back(st.X[a]); nw.m.push_back(st.m[a]); nw.s2.push_back(st.s2[a]);
        }
        st = std::move(nw);
    }
    return changed;
}
