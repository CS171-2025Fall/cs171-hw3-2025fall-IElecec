#include "rdr/bdpt.h"
#include "rdr/guided.h"
#include "rdr/integrator.h"
#include "rdr/photon.h"

RDR_NAMESPACE_BEGIN

RDR_REGISTER_FACTORY(Integrator, [](const Properties &props) -> Integrator * {
  auto type = props.getProperty<std::string>("type", "path");
  if (type == "intersection_test") {
    return Memory::alloc<IntersectionTestIntegrator>(props);
  } 
  if (type == "area_light") {
    return Memory::alloc<AreaLightIntegrator>(props);
  }
  
  Exception_("Integrator type {} not found", type);

  return nullptr;
})

RDR_NAMESPACE_END
