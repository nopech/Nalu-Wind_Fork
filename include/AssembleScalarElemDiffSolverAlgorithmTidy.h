/* 
 * File:   AssembleScalarElemDiffSolverAlgorithmTidy.h
 * Author: Raphael Lindegger
 *
 * Created on September 25, 2019, 10:58 AM
 */

#ifndef ASSEMBLESCALARELEMDIFFSOLVERALGORITHMTIDY_H
#define ASSEMBLESCALARELEMDIFFSOLVERALGORITHMTIDY_H

#include <SolverAlgorithm.h>
#include <FieldTypeDef.h>

namespace sierra{
namespace nalu{

class stk::mesh::Part;
class Realm;

/** Solver algorithm to compute the coefficients of the
 * diffusion equation.
 */
class AssembleScalarElemDiffSolverAlgorithmTidy : public SolverAlgorithm {
public:
    AssembleScalarElemDiffSolverAlgorithmTidy(Realm &realm,
                                          stk::mesh::Part *part,
                                          EquationSystem *eqSystem,
                                          ScalarFieldType *scalarQ,
                                          VectorFieldType *dqdx,
                                          ScalarFieldType *diffFluxCoeff);
    virtual ~AssembleScalarElemDiffSolverAlgorithmTidy() {}
    virtual void initialize_connectivity();
    virtual void execute();

private:
    ScalarFieldType * scalarQ_;
    ScalarFieldType * diffFluxCoeff_;
    VectorFieldType * coordinates_;
    const bool shiftedGradOp_;
};

} // namespace nalu
} // namespace Sierra

#endif /* ASSEMBLESCALARELEMDIFFSOLVERALGORITHMTIDY_H */

