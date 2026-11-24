#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <AMReX_Print.H>

#include "CGTable.H"

using namespace amrex;

namespace {

// the same U as the prototype's Kernel.U: not truncated by ip_range
Real pot_U (Real r, PotentialParams const& pot) { return ip_U(r, pot); }

}

void
CGTable::build (PotentialParams const& pot, Real rmax, int nr, Real s2min, Real s2max, int ns, Real rquad)
{
    m_nr = nr;
    m_ns = ns;
    m_r0 = 0.0;
    m_dr = rmax/(nr - 1);
    m_ls0 = std::log(s2min);
    m_dls = (std::log(s2max) - std::log(s2min))/(ns - 1);
    m_W.assign(std::size_t(ns)*nr, 0.0);
    m_Wr.assign(std::size_t(ns)*nr, 0.0);
    m_Ws.assign(std::size_t(ns)*nr, 0.0);

    // Uhat(0) = 4 pi int_0^0.6 U r^2 dr by the trapezoidal rule on 60001 points, plus the local repulsion
    {
        const int n = 60001;
        const Real h = 0.6/(n - 1);
        Real s = 0.0;
        for (int q = 0; q < n; ++q) {
            const Real r = q*h;
            const Real f = pot_U(r, pot)*r*r;
            s += (q == 0 || q == n - 1) ? 0.5*f : f;
        }
        uhat0 = 4.0*M_PI*s*h + pot.ip_local_rep;
    }

    const Real twopi = 2.0*M_PI;
    std::vector<Real> rp, gV;
    for (int is = 0; is < ns; ++is) {
        const Real s2 = std::exp(m_ls0 + is*m_dls);
        const Real s = std::sqrt(s2);
        const Real h = std::min(s/8.0, 2.5e-4);
        // quadrature nodes r' = 0, h, 2h, ... < rquad + h (numpy arange), trapezoidal weights
        const int nq = static_cast<int>(std::ceil((rquad + h)/h));
        rp.resize(nq);
        gV.resize(nq);
        for (int q = 0; q < nq; ++q) {
            rp[q] = q*h;
            const Real wq = (q == 0 || q == nq - 1) ? 0.5*h : h;
            gV[q] = rp[q]*pot_U(rp[q], pot)*wq;
        }
        const Real pref = 1.0/(s*std::sqrt(twopi));
        for (int ir = 0; ir < nr; ++ir) {
            const Real r = m_r0 + ir*m_dr;
            Real W = 0.0, Wr = 0.0, Ws = 0.0;
            for (int q = 0; q < nq; ++q) {
                const Real um = r - rp[q], up = r + rp[q];
                const Real em = std::exp(-um*um/(2.0*s2));
                const Real ep = std::exp(-up*up/(2.0*s2));
                W  += gV[q]*(em - ep);
                Wr += gV[q]*(-um/s2*em + up/s2*ep);
                Ws += gV[q]*((um*um/(2.0*s2*s2) - 1.0/(2.0*s2))*em - (up*up/(2.0*s2*s2) - 1.0/(2.0*s2))*ep);
            }
            W *= pref; Wr *= pref; Ws *= pref;
            const std::size_t k = std::size_t(is)*nr + ir;
            if (ir == 0) {
                // r = 0 limits: W(0) = (2 pi s2)^{-3/2} 4 pi int r'^2 U e^{-r'^2/2 s2}; W'(0) = 0
                Real w0 = 0.0, ws0 = 0.0;
                for (int q = 0; q < nq; ++q) {
                    const Real e0 = std::exp(-rp[q]*rp[q]/(2.0*s2));
                    w0  += gV[q]*rp[q]*e0;
                    ws0 += gV[q]*rp[q]*e0*(rp[q]*rp[q]/(2.0*s2*s2) - 1.5/s2);
                }
                const Real norm = std::pow(twopi*s2, 1.5);
                m_W[k]  = 4.0*M_PI*w0/norm;
                m_Ws[k] = 4.0*M_PI*ws0/norm;
                m_Wr[k] = 0.0;
            } else {
                m_W[k]  = W/r;
                m_Ws[k] = Ws/r;
                m_Wr[k] = Wr/r - W/(r*r);
            }
        }
    }
    finish();
}

void
CGTable::finish ()
{
    // d(dW/ds2)/dr at the nodes by centred differences (one-sided at the ends), as numpy.gradient
    m_Wsr.assign(m_W.size(), 0.0);
    for (int is = 0; is < m_ns; ++is) {
        const std::size_t o = std::size_t(is)*m_nr;
        for (int ir = 0; ir < m_nr; ++ir) {
            if (ir == 0) {
                m_Wsr[o] = (m_Ws[o+1] - m_Ws[o])/m_dr;
            } else if (ir == m_nr - 1) {
                m_Wsr[o+ir] = (m_Ws[o+ir] - m_Ws[o+ir-1])/m_dr;
            } else {
                m_Wsr[o+ir] = (m_Ws[o+ir+1] - m_Ws[o+ir-1])/(2.0*m_dr);
            }
        }
    }
    m_s2.resize(m_ns);
    for (int is = 0; is < m_ns; ++is) { m_s2[is] = std::exp(m_ls0 + is*m_dls); }
}

bool
CGTable::read (const std::string& file)
{
    std::FILE* f = std::fopen(file.c_str(), "rb");
    if (!f) { return false; }
    std::int64_t hdr[2];
    double par[5];
    bool ok = std::fread(hdr, sizeof(std::int64_t), 2, f) == 2 && std::fread(par, sizeof(double), 5, f) == 5;
    if (ok) {
        m_nr = static_cast<int>(hdr[0]);
        m_ns = static_cast<int>(hdr[1]);
        m_r0 = par[0]; m_dr = par[1]; m_ls0 = par[2]; m_dls = par[3]; uhat0 = par[4];
        const std::size_t n = std::size_t(m_nr)*m_ns;
        m_W.resize(n); m_Wr.resize(n); m_Ws.resize(n);
        ok = std::fread(m_W.data(), sizeof(double), n, f) == n
          && std::fread(m_Wr.data(), sizeof(double), n, f) == n
          && std::fread(m_Ws.data(), sizeof(double), n, f) == n;
    }
    std::fclose(f);
    if (!ok) { m_nr = m_ns = 0; return false; }
    finish();
    return true;
}

void
CGTable::write (const std::string& file) const
{
    std::FILE* f = std::fopen(file.c_str(), "wb");
    if (!f) { amrex::Print() << "CGTable: cannot write " << file << "\n"; return; }
    const std::int64_t hdr[2] = {m_nr, m_ns};
    const double par[5] = {m_r0, m_dr, m_ls0, m_dls, uhat0};
    std::fwrite(hdr, sizeof(std::int64_t), 2, f);
    std::fwrite(par, sizeof(double), 5, f);
    const std::size_t n = m_W.size();
    std::fwrite(m_W.data(), sizeof(double), n, f);
    std::fwrite(m_Wr.data(), sizeof(double), n, f);
    std::fwrite(m_Ws.data(), sizeof(double), n, f);
    std::fclose(f);
}

void
CGTable::eval (Real r, Real s2, Real& W, Real& Wr, Real& Ws) const
{
    const Real rm = rmax();
    if (r >= rm) { W = Wr = Ws = 0.0; return; }
    // widths outside the table are clamped
    const Real ls = std::log(std::min(std::max(s2, m_s2.front()), m_s2.back()));
    int js = static_cast<int>((ls - m_ls0)/m_dls);
    js = std::min(std::max(js, 0), m_ns - 2);
    const Real u = (ls - (m_ls0 + js*m_dls))/m_dls;
    const Real rc = std::min(r, rm - 1.e-12);
    int ir = static_cast<int>(rc/m_dr);
    ir = std::min(std::max(ir, 0), m_nr - 2);
    const Real t = (rc - (m_r0 + ir*m_dr))/m_dr;
    const Real h = m_dr;

    const Real t2 = t*t, t3 = t2*t;
    const Real h00 = 2*t3 - 3*t2 + 1, h10 = t3 - 2*t2 + t, h01 = -2*t3 + 3*t2, h11 = t3 - t2;
    const Real d00 = 6*t2 - 6*t, d10 = 3*t2 - 4*t + 1, d01 = -6*t2 + 6*t, d11 = 3*t2 - 2*t;

    auto herm = [&] (int j, Real& v, Real& dv) {
        const std::size_t k = std::size_t(j)*m_nr + ir;
        const Real w0 = m_W[k], w1 = m_W[k+1], m0 = m_Wr[k], m1 = m_Wr[k+1];
        v  = h00*w0 + h10*h*m0 + h01*w1 + h11*h*m1;
        dv = (d00*w0 + d10*h*m0 + d01*w1 + d11*h*m1)/h;
    };
    Real v0, dv0, v1, dv1;
    herm(js, v0, dv0);
    herm(js + 1, v1, dv1);
    const std::size_t k0 = std::size_t(js)*m_nr + ir, k1 = k0 + m_nr;
    // slopes in ln s2: s2 dW/ds2, linear in r between the nodes
    const Real s0 = m_s2[js]*(m_Ws[k0]*(1 - t) + m_Ws[k0+1]*t);
    const Real s1 = m_s2[js+1]*(m_Ws[k1]*(1 - t) + m_Ws[k1+1]*t);
    const Real u2 = u*u, u3 = u2*u;
    const Real g00 = 2*u3 - 3*u2 + 1, g10 = u3 - 2*u2 + u, g01 = -2*u3 + 3*u2, g11 = u3 - u2;
    const Real e00 = 6*u2 - 6*u, e10 = 3*u2 - 4*u + 1, e01 = -6*u2 + 6*u, e11 = 3*u2 - 2*u;
    const Real hl = m_dls;
    W = g00*v0 + g10*hl*s0 + g01*v1 + g11*hl*s1;
    const Real q0 = m_s2[js]*(m_Wsr[k0]*(1 - t) + m_Wsr[k0+1]*t);
    const Real q1 = m_s2[js+1]*(m_Wsr[k1]*(1 - t) + m_Wsr[k1+1]*t);
    Wr = g00*dv0 + g10*hl*q0 + g01*dv1 + g11*hl*q1;
    const Real dWdls = (e00*v0 + e10*hl*s0 + e01*v1 + e11*hl*s1)/hl;
    Ws = dWdls/s2;
}
