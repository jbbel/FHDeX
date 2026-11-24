#ifndef CG_MODEL_H_
#define CG_MODEL_H_

#include <array>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "CGTable.H"

// The cluster-level model on plain arrays (every rank holds the whole, small, set of clusters;
// see ClusterDriver). Ported one to one from the numpy prototype gema_patterns/scripts/cgmodel/
// (model.py, integrate.py, neighbors.py): the same rules give the same trajectory to round-off.
//
// State: centres X in the periodic unit box, masses m (sum 1), variances s2, ids (the DK ids).
// Ingredients: smoothed potential W(r; s2) (CGTable); forces -grad F_cg; widths by the cage
// rule s2 = 3 D / Lap Phi(X_j) iterated; mu_j = D [ln m_j - 3/2 ln(2 pi s2_j)] - D/2 + Phi(X_j);
// Gabriel neighbours; saddles of Phi between neighbours by segment sampling and Newton steps;
// exchange J_{j->k} = f_s(b) A exp(-(U_s - mu_j)/D) with f_s(b) = fs exp(-gamma (b - b0)),
// b = (U_s - mu_j)/D, applied antisymmetrically; events: merge and dissolve.

using Vec3 = std::array<amrex::Real, 3>;

struct CGParams
{
    amrex::Real D = 0.30;
    amrex::Real N = 20971520.0;
    amrex::Real fs = 0.28;          // flux factor at barrier b0
    amrex::Real gamma = 0.3;        // f_s(b) = fs exp(-gamma (b - b0))
    amrex::Real b0 = 6.0;
    amrex::Real rcut = 0.32;        // potential cutoff
    amrex::Real r_nb = 0.25;        // candidate neighbour cutoff before the Gabriel test
    amrex::Real gabriel_beta = 1.0;
    int nseg = 9;                   // segment samples for the saddle search
    int newton = 6;                 // Newton iterations on grad Phi (a third when warm-started)
    int saddle_int = 5;             // recompute saddles every this many steps (warm-started)
    int rebuild_int = 10;           // rebuild pair lists every this many steps
    amrex::Real barrier_min = 0.5;  // channels with U_s < max(mu) + barrier_min D are not activated
    amrex::Real dm_cap = 0.2;       // max fraction of the smaller mass exchanged by one pair per step
    amrex::Real dt_kappa = 0.5;     // adaptive step dt <= dt_kappa / stiffness
    int width_iters = 3;
    amrex::Real m_min = 7.e-5;      // dissolution threshold (DK detection equivalent)
    amrex::Real merge_frac = 0.5;   // merge when r < merge_frac (sigma_i + sigma_j) ...
    amrex::Real merge_rmax = 0.08;  // ... or r < merge_rmax with no barrier between the pair
    amrex::Real wide_frac = 0.35;   // dissolve when sigma > wide_frac x nearest distance
    bool mass_noise = false;
    bool pos_noise = false;
    bool spill = true;              // channels without an accepted barrier carry the floor flux down a mu difference > barrier_min D
    unsigned long seed = 0;
};

struct CGState
{
    std::vector<long> ids;
    std::vector<Vec3> X;
    std::vector<amrex::Real> m, s2;
    amrex::Real t = 0.0;
    long step = 0;
    int size () const { return static_cast<int>(m.size()); }
};

struct CGPairs
{
    std::vector<int> i, j;            // i < j, all pairs within rcut
    std::vector<Vec3> d;              // X_j - X_i (minimum image) at build time
    std::vector<amrex::Real> r;
    std::vector<char> gab;            // Gabriel pair
    std::vector<std::vector<int>> adj;  // neighbour lists (all rcut pairs)
    int ngab = 0;
};

struct CGSaddles
{
    std::vector<int> sel, i, j;       // pair index and members
    std::vector<Vec3> xs;             // saddle position
    std::vector<amrex::Real> Us, A, gnorm;
    std::vector<amrex::Real> Useg;    // maximum of Phi along the straight segment (ridge height; always computed)
    std::vector<Vec3> lam;            // Hessian eigenvalues, ascending
    std::vector<char> ok;
};

struct CGEvent
{
    amrex::Real t;
    long step;
    std::string type;                 // "merge" or "dissolve"
    std::vector<long> old_ids, new_ids;
    std::vector<amrex::Real> old_m, new_m;
};

class CGModel
{
public:

    CGModel (CGTable const& tab, CGParams const& p) : m_tab(tab), m_p(p), m_rng(p.seed) {}

    CGParams const& params () const { return m_p; }
    CGTable const& table () const { return m_tab; }

    // ---- geometry
    static Vec3 min_image (Vec3 const& d);
    static amrex::Real norm (Vec3 const& d);
    void build_pairs (CGState const& st, CGPairs& pr) const;

    // ---- statics
    void widths (CGState& st) const;                                   // cage rule, iterated
    void mu (CGState const& st, CGPairs const& pr, std::vector<amrex::Real>& mu) const;
    void forces (CGState const& st, std::vector<Vec3>& f) const;
    // F_cg and its parts (entropy, self, pair)
    amrex::Real free_energy (CGState const& st, amrex::Real& f_ent, amrex::Real& f_self, amrex::Real& f_pair) const;
    amrex::Real kappa_X (CGState const& st, CGPairs const& pr) const;  // position stiffness estimate

    // ---- saddles and exchange
    void saddles (CGState const& st, CGPairs const& pr, CGSaddles& sad, const std::vector<Vec3>* warm) const;
    // dm from the transition-state fluxes over dt; Jij (i -> j) and Jji per Gabriel pair
    void exchange (CGState const& st, std::vector<amrex::Real> const& mu, CGSaddles const& sad, amrex::Real dt,
                   std::vector<amrex::Real>& dm, std::vector<amrex::Real>& Jij, std::vector<amrex::Real>& Jji);
    int n_capped () const { return m_ncapped; }
    int n_spill () const { return m_nspill; }

    // ---- events (mergers, then dissolutions); returns true if the cluster set changed
    bool handle_events (CGState& st, CGPairs const& pr, CGSaddles const& sad, std::vector<amrex::Real> const& mu,
                        std::vector<amrex::Real> const& Jij, std::vector<amrex::Real> const& Jji,
                        std::vector<CGEvent>& events, std::function<void(const std::string&)> const& log);

    // field of the single-smoothed Gaussians at a point from a contributor list: Phi, grad, Hessian
    void field (Vec3 const& x, CGState const& st, std::vector<int> const& contrib, int order,
                amrex::Real& phi, Vec3& g, std::array<amrex::Real, 9>& H) const;

    amrex::Real normal () { return m_normal(m_rng); }

private:

    CGTable const& m_tab;
    CGParams m_p;
    std::mt19937_64 m_rng;
    std::normal_distribution<amrex::Real> m_normal {0.0, 1.0};
    int m_ncapped = 0;
    int m_nspill = 0;
};

// eigenvalues of a symmetric 3x3 matrix (row-major, 9 entries), ascending
void eig3_sym (std::array<amrex::Real, 9> const& H, Vec3& lam);
// solve (H + eps I) x = b for a 3x3 matrix
Vec3 solve3 (std::array<amrex::Real, 9> H, Vec3 const& b, amrex::Real eps);

#endif
