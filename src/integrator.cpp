#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

#include <iostream>

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        //
        //
        // Accumulate radiance
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      //
      // Useful Functions:
      // @see BSDF::sample
      // @see SurfaceInteraction::spawnRay
      //
      // You should update ray = ... with the spawned ray
      float pdf;
      interaction.bsdf->sample(interaction, sampler, &pdf);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0, 0, 0);
  Float dist_to_light = Norm(point_light_position - interaction.p);
  Vec3f light_dir     = Normalize(point_light_position - interaction.p);
  auto test_ray       = DifferentialRay(interaction.p, light_dir);

  // TODO(HW3): Test for occlusion
  //
  // You should test if there is any intersection between interaction.p and
  // point_light_position using scene->intersect. If so, return an occluded
  // color. (or Vec3f color(0, 0, 0) to be specific)
  //
  // You may find the following variables useful:
  //
  // @see bool Scene::intersect(const Ray &ray, SurfaceInteraction &interaction)
  //    This function tests whether the ray intersects with any geometry in the
  //    scene. And if so, it returns true and fills the interaction with the
  //    intersection information.
  //
  //    You can use iteraction.p to get the intersection position.
  //
  SurfaceInteraction occluded_intersection;
  if (scene->intersect(test_ray, occluded_intersection)){
    if (Norm(occluded_intersection.p - interaction.p)< dist_to_light)
      return Vec3f(0, 0, 0);
  }

  // Not occluded, compute the contribution using perfect diffuse diffuse model
  // Perform a quick and dirty check to determine whether the BSDF is ideal
  // diffuse by RTTI
  const BSDF *bsdf      = interaction.bsdf;
  bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

  if (bsdf != nullptr && is_ideal_diffuse) {
    // TODO(HW3): Compute the contribution
    //
    // You can use bsdf->evaluate(interaction) * cos_theta to approximate the
    // albedo. In this homework, we do not need to consider a
    // radiometry-accurate model, so a simple phong-shading-like model is can be
    // used to determine the value of color.

    // The angle between light direction and surface normal
    Float cos_theta =
        std::max(Dot(light_dir, interaction.normal), 0.0f);  // one-sided

    // You should assign the value to color
    Float pi = M_PI;
    color = bsdf->evaluate(interaction) * cos_theta * point_light_flux / (pi * 4.0f  *dist_to_light * dist_to_light);
  }

  return color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * Area Light Integrator's Implementation
 *
 * ===================================================================== */

void AreaLightIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
   // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {

        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f AreaLightIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    bool is_area_light = 
    interaction.primitive->getAreaLight() != nullptr;
    if(is_area_light){
      color = interaction.primitive->getAreaLight()->Le(interaction,interaction.wo);
      return color;
    }

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      float pdf;
      interaction.bsdf->sample(interaction, sampler, &pdf);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f AreaLightIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0, 0, 0);
  
  const int sample_number = 100;
  for (const ref<Light> &light : scene->getLights()){
    if(dynamic_cast<AreaLight*>(light.get()) == nullptr) continue;
    /*
    #pragma omp parallel
    {
      Vec3f local_color(0,0,0);
      #pragma omp for
      for(int i = 0;i < sample_number;i++){
        Sampler sampler;

        SurfaceInteraction light_interation = light->sample(sampler);
        Vec3f point_light_position = light_interation.p; 
        
        Float dist_to_light = Norm(point_light_position - interaction.p);
        Vec3f light_dir     = Normalize(point_light_position - interaction.p);

        Vec3f point_light_flux = light->Le(light_interation,-light_dir);

        auto test_ray       = DifferentialRay(interaction.p, light_dir);
        SurfaceInteraction occlued_intersection;
        if (scene->intersect(test_ray, occlued_intersection)){
          if (Norm(occlued_intersection.p - interaction.p)< dist_to_light){
            // color += Vec3f(0, 0, 0);
            continue;
          }
        }

        // Not occluded, compute the contribution using perfect diffuse diffuse model
        // Perform a quick and dirty check to determine whether the BSDF is ideal
        // diffuse by RTTI
        const BSDF *bsdf      = interaction.bsdf;
        bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

        if (bsdf != nullptr && is_ideal_diffuse) {

          Float cos_theta =
              std::max(Dot(light_dir, interaction.normal), 0.0f);  // one-sided

          // You should assign the value to color
          local_color += bsdf->evaluate(interaction) * cos_theta * point_light_flux / (sample_number*1.0f);
        }
      }
      #pragma omp critical
        color += local_color;
    }
    */
    Sampler sampler;
    for(int i = 0;i < sample_number;i++){
      SurfaceInteraction light_interation = light->sample(sampler);
      Vec3f point_light_position = light_interation.p; 
      
      Float dist_to_light = Norm(point_light_position - interaction.p);
      Vec3f light_dir     = Normalize(point_light_position - interaction.p);

      //two-sided
      // Vec3f point_light_flux = std::max(light->Le(light_interation,light_dir),
      // light->Le(light_interation,-light_dir));
      //one-sided
      Vec3f point_light_flux = light->Le(light_interation,-light_dir);

      auto test_ray       = DifferentialRay(interaction.p, light_dir);
      SurfaceInteraction occluded_intersection;
      if (scene->intersect(test_ray, occluded_intersection)){
        if (Norm(occluded_intersection.p - interaction.p)< dist_to_light){
          // color += Vec3f(0, 0, 0);
          continue;
        }
      }

      // Not occluded, compute the contribution using perfect diffuse diffuse model
      // Perform a quick and dirty check to determine whether the BSDF is ideal
      // diffuse by RTTI
      const BSDF *bsdf      = interaction.bsdf;
      bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

      if (bsdf != nullptr && is_ideal_diffuse) {

        Float cos_theta =
            std::max(Dot(light_dir, interaction.normal), 0.0f);  // one-sided

        // You should assign the value to color
        Float pi = M_PI;
        color += bsdf->evaluate(interaction) * cos_theta * point_light_flux / (pi * 4.0f *dist_to_light * dist_to_light) / (sample_number*1.0f);
        // color += bsdf->evaluate(interaction) * cos_theta / (sample_number*1.0f);
      }
    }
  }
  return color;
}

/* ===================================================================== *
 *
 * Env Light Integrator's Implementation
 *
 * ===================================================================== */

void EnvLightIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {

        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f EnvLightIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      float pdf;
      interaction.bsdf->sample(interaction, sampler, &pdf);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f EnvLightIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f color(0, 0, 0);
  
  const int sample_number = 100;

  Sampler sampler;
  const ref<InfiniteAreaLight> light = scene->getInfiniteLight();
  for(int i = 0;i < sample_number;i++){
    SurfaceInteraction temp_interaction = interaction;
    SurfaceInteraction light_interation = light->sample(temp_interaction,sampler);

    Vec3f wi = temp_interaction.wi;

    Vec3f radiance = light->Le(light_interation, wi);

    auto test_ray = interaction.spawnRay(wi);
    SurfaceInteraction occluded_intersection;
    if (scene->intersect(test_ray, occluded_intersection)){
        continue;
    }

    // Not occluded, compute the contribution using perfect diffuse diffuse model
    // Perform a quick and dirty check to determine whether the BSDF is ideal
    // diffuse by RTTI
    const BSDF *bsdf      = interaction.bsdf;
    bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

    if (bsdf != nullptr && is_ideal_diffuse) {

      Float cos_theta =
          std::max(Dot(wi, interaction.normal), 0.0f);  // one-sided

      // You should assign the value to color
      color += bsdf->evaluate(interaction) * cos_theta * radiance / light_interation.pdf / (sample_number*1.0f);
      // color += bsdf->evaluate(interaction) * cos_theta / (sample_number*1.0f);
    }
  }
  return color;
}

RDR_NAMESPACE_END
