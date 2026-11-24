#ifndef CG_TABLE_H_
#define CG_TABLE_H_

#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "Potentials.H"

// W(r; s2) = (U * G_{s2})(r): the pair potential smoothed by an isotropic Gaussian of
// variance s2 per coordinate, with dW/dr and dW/ds2, tabulated on r (uniform) x ln s2
// (uniform). Evaluation is cubic Hermite in r (slopes dW/dr) and cubic Hermite in ln s2
// (slopes s2 dW/ds2), exactly as the prototype's WTable (gema_patterns/scripts/pm/
// dk_cgmodel.py), so that the two codes see the same potential to round-off. Because
// Gaussian smoothing obeys the heat equation, the Laplacian of W in r is 2 dW/ds2.
class CGTable
{
public:

    // Build the table from the potential by the radial quadrature of the prototype.
    void build (PotentialParams const& pot, amrex::Real rmax = 0.35, int nr = 701,
                amrex::Real s2min = 5.e-6, amrex::Real s2max = 6.e-3, int ns = 112,
                amrex::Real rquad = 0.5);

    // Binary cache: header (2 int64: nr, ns; 5 doubles: r0, dr, ln s2_0, d ln s2, uhat0),
    // then W, Wr, Ws as [ns][nr] doubles. read returns false if the file is missing.
    bool read (const std::string& file);
    void write (const std::string& file) const;

    // W, dW/dr, dW/ds2 at (r, s2): zero beyond rmax, s2 clamped to the table range.
    void eval (amrex::Real r, amrex::Real s2, amrex::Real& W, amrex::Real& Wr, amrex::Real& Ws) const;

    amrex::Real W (amrex::Real r, amrex::Real s2) const
    {
        amrex::Real w, wr, ws;
        eval(r, s2, w, wr, ws);
        return w;
    }

    // Laplacian of W in r
    amrex::Real lap (amrex::Real r, amrex::Real s2) const
    {
        amrex::Real w, wr, ws;
        eval(r, s2, w, wr, ws);
        return 2.0*ws;
    }

    amrex::Real rmax () const { return m_r0 + (m_nr - 1)*m_dr; }
    amrex::Real s2min () const { return std::exp(m_ls0); }
    amrex::Real s2max () const { return std::exp(m_ls0 + (m_ns - 1)*m_dls); }
    int nr () const { return m_nr; }
    int ns () const { return m_ns; }
    bool empty () const { return m_nr == 0; }

    // 4 pi int U r^2 dr (+ the local repulsion): Uhat(0), so that F_uni = uhat0/2
    amrex::Real uhat0 = 0.0;

private:

    void finish ();

    int m_nr = 0, m_ns = 0;
    amrex::Real m_r0 = 0.0, m_dr = 0.0, m_ls0 = 0.0, m_dls = 0.0;
    std::vector<amrex::Real> m_W, m_Wr, m_Ws, m_Wsr, m_s2;
};

#endif
