/* 
 * File:   LaplaceValidationAuxFunction.C
 * Author: Raphael Lindegger
 * 
 * Created on January 4, 2020, 2:09 PM
 */

#include <user_functions/LaplaceValidationAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

LaplaceValidationAuxFunction::LaplaceValidationAuxFunction() :
  AuxFunction(0, 1),
  pi_(std::acos(-1.0))
{
  // Nothing to do
}

void
LaplaceValidationAuxFunction::do_evaluate(
  const double * coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for(unsigned p=0; p < numPoints; ++p) {
    
    if (spatialDimension == 2) {
      const int offset = 2*p;
      const double cX = coords[offset + 0];
      const double cY = coords[offset + 1];
      const double b = 2.0/3.0 * pi_;

      fieldPtr[p] = sin(b*cX)*sinh(b*cY);
    }
    else if (spatialDimension == 3) {
    }
  }
}

} // namespace nalu
} // namespace Sierra