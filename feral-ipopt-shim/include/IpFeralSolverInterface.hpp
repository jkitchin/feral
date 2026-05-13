// FeralSolverInterface — Ipopt linear-solver plug-in for feral.
//
// Subclasses Ipopt's SparseSymLinearSolverInterface and forwards
// each method to feral's C ABI (feral_capi.h).
//
// Matrix format: CSR_Format_0_Offset. For a symmetric matrix this
// is byte-identical to feral's CscMatrix layout (upper-tri CSR
// ≡ lower-tri CSC).
//
// POC scope: no option forwarding (uses Solver::new() defaults),
// no IncreaseQuality (returns false), no SYMSOLVER_CALL_AGAIN.
//
// Lives in the Ipopt namespace alongside the other linear-solver
// interfaces; built into libipopt by the feral-ipopt-shim patch.

#ifndef __IPFERALSOLVERINTERFACE_HPP__
#define __IPFERALSOLVERINTERFACE_HPP__

#include "IpSparseSymLinearSolverInterface.hpp"

extern "C" {
#include "feral_capi.h"
}

namespace Ipopt
{

class FeralSolverInterface : public SparseSymLinearSolverInterface
{
public:
   FeralSolverInterface();
   virtual ~FeralSolverInterface();

   bool InitializeImpl(
      const OptionsList& options,
      const std::string& prefix
   );

   virtual ESymSolverStatus InitializeStructure(
      Index        dim,
      Index        nonzeros,
      const Index* ia,
      const Index* ja
   );

   virtual Number* GetValuesArrayPtr();

   virtual ESymSolverStatus MultiSolve(
      bool         new_matrix,
      const Index* ia,
      const Index* ja,
      Index        nrhs,
      Number*      rhs_vals,
      bool         check_NegEVals,
      Index        numberOfNegEVals
   );

   virtual Index NumberOfNegEVals() const;

   virtual bool IncreaseQuality();

   virtual bool ProvidesInertia() const
   {
      return true;
   }

   EMatrixFormat MatrixFormat() const
   {
      return CSR_Format_0_Offset;
   }

   static void RegisterOptions(
      SmartPtr<RegisteredOptions> /*roptions*/
   )
   {
      // POC: no feral-specific options registered.
   }

   static std::string GetName()
   {
      return "feral";
   }

private:
   FeralSolverInterface(const FeralSolverInterface&);
   void operator=(const FeralSolverInterface&);

   FeralSolver* solver_;
   Index        dim_;
   Index        nonzeros_;
   bool         have_structure_;
   bool         have_factor_;
   Index        negevals_;
};

} // namespace Ipopt

#endif
