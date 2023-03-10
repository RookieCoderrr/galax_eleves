#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

//#define GALAX_OMP_VERSION 
#define GALAX_OMP_SIMD_VERSION 

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;


Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}
#pragma omp declare reduction(vec_float_plus                                                                                         \
                             : std::vector <float>                                                                                   \
                             : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus <float>())) \
   initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
void Model_CPU_fast
::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

// OMP  version
#ifdef GALAX_OMP_VERSION
    #pragma omp parallel for schedule(dynamic, 100)  reduction(vec_float_plus \
                                  : accelerationsx, accelerationsy, accelerationsz)

	for (int i = 0; i < n_particles; i++)
	{
		#pragma omp simd
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}


    auto& particlesx = particles.x;
    auto& particlesy = particles.y;
    auto& particlesz = particles.z;
    
    #pragma omp parallel for schedule(dynamic, 100)  reduction(vec_float_plus \
                                  : velocitiesx, velocitiesy, velocitiesz, particlesx, particlesy, particlesz)
	//#pragma omp for simd
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particlesx[i] += velocitiesx   [i] * 0.1f;
		particlesy[i] += velocitiesy   [i] * 0.1f;
		particlesz[i] += velocitiesz   [i] * 0.1f;
	}
	
#endif


#ifdef GALAX_OMP_SIMD_VERSION

// OMP + xsimd version
#pragma omp parallel for
     for (int i = 0; i < n_particles; i += b_type::size)
     {
         // load registers body i
         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
         const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
         const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        // calculate force
		for (int j = 0; j < n_particles; j+=b_type::size)
		{
			 // load registers body j
         	const b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
         	const b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
         	const b_type rposz_j = b_type::load_unaligned(&particles.z[j]);

			
			const b_type diffx = rposx_j - rposx_i;
			const b_type diffy = rposy_j - rposy_i;
			const b_type diffz = rposz_j - rposz_i;

			b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

			const auto comp = xs::lt(dij, b_type(1.0f));
			dij = xs::select(comp, b_type(10.0f), b_type(10.0f) / xs::pow(dij, b_type(3.0f/2.0f)));
			/*if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = std::sqrt(dij);
				dij = 10.0 / (dij * dij * dij);
			}*/
			// update accelaration on i
			raccx_i += diffx * dij * initstate.masses[j];
			raccy_i += diffy * dij * initstate.masses[j];
			raccz_i += diffz * dij * initstate.masses[j];
			
		}

		
		// load register into memory
		raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);

     }
	auto& particlesx = particles.x;
    auto& particlesy = particles.y;
    auto& particlesz = particles.z;
	#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		b_type rposx_i = b_type::load_unaligned(&particlesx[i]);
        b_type rposy_i = b_type::load_unaligned(&particlesy[i]);
        b_type rposz_i = b_type::load_unaligned(&particlesz[i]);
		b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
		b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
		b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
		const b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        const b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        const b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

		rvelx_i += raccx_i * 2.0f;
		rvely_i += raccy_i * 2.0f;
		rvelz_i += raccz_i * 2.0f;
		rposx_i += rvelx_i * 0.1f;
		rposy_i += rvely_i * 0.1f;
		rposz_i += rvelz_i * 0.1f;

		rvelx_i.store_unaligned(&velocitiesx[i]);
		rvely_i.store_unaligned(&velocitiesy[i]);
		rvelz_i.store_unaligned(&velocitiesz[i]);
		rposx_i.store_unaligned(&particlesx[i]);
		rposy_i.store_unaligned(&particlesy[i]);
		rposz_i.store_unaligned(&particlesz[i]);
	}
#endif

}

#endif // GALAX_MODEL_CPU_FAST
