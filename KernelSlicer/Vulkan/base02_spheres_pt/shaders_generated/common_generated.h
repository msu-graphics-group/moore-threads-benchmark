/////////////////////////////////////////////////////////////////////
/////////////  Required  Shader Features ////////////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
#ifndef uint32_t
#define uint32_t uint
#endif
#define FLT_MAX 1e37f
#define FLT_MIN -1e37f
#define FLT_EPSILON 1e-6f
#define DEG_TO_RAD  0.017453293f
#define unmasked
#define half  float16_t
#define half2 f16vec2
#define half3 f16vec3
#define half4 f16vec4
bool  isfinite(float x)            { return !isinf(x); }
float copysign(float mag, float s) { return abs(mag)*sign(s); }

struct complex
{
  float re, im;
};

complex make_complex(float re, float im) { 
  complex res;
  res.re = re;
  res.im = im;
  return res;
}

complex to_complex(float re)              { return make_complex(re, 0.0f);}
complex complex_add(complex a, complex b) { return make_complex(a.re + b.re, a.im + b.im); }
complex complex_sub(complex a, complex b) { return make_complex(a.re - b.re, a.im - b.im); }
complex complex_mul(complex a, complex b) { return make_complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
complex complex_div(complex a, complex b) {
  const float scale = 1 / (b.re * b.re + b.im * b.im);
  return make_complex(scale * (a.re * b.re + a.im * b.im), scale * (a.im * b.re - a.re * b.im));
}

complex real_add_complex(float value, complex z) { return complex_add(to_complex(value),z); }
complex real_sub_complex(float value, complex z) { return complex_sub(to_complex(value),z); }
complex real_mul_complex(float value, complex z) { return complex_mul(to_complex(value),z); }
complex real_div_complex(float value, complex z) { return complex_div(to_complex(value),z); }

complex complex_add_real(complex z, float value) { return complex_add(z, to_complex(value)); }
complex complex_sub_real(complex z, float value) { return complex_sub(z, to_complex(value)); }
complex complex_mul_real(complex z, float value) { return complex_mul(z, to_complex(value)); }
complex complex_div_real(complex z, float value) { return complex_div(z, to_complex(value)); }

float real(complex z) { return z.re;}
float imag(complex z) { return z.im; }
float complex_norm(complex z) { return z.re * z.re + z.im * z.im; }
float complex_abs(complex z) { return sqrt(complex_norm(z)); }
complex complex_sqrt(complex z) 
{
  float n = complex_abs(z);
  float t1 = sqrt(0.5f * (n + abs(z.re)));
  float t2 = 0.5f * z.im / t1;
  if (n == 0.0f)
    return to_complex(0.0f);
  if (z.re >= 0.0f)
    return make_complex(t1, t2);
  else
    return make_complex(abs(t2), copysign(t1, z.im));
}

struct Lite_Hit
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
};
const float GEPSILON = 5e-6f;
const float DEPSILON = 1e-20f;
const uint THREAD_IS_DEAD = 2147483648;
struct SphereMaterial
{
  vec4 color;
};
struct RandomGen
{
  uvec2 state;
};

#define M_PI          3.14159265358979323846f
#define M_TWOPI       6.28318530717958647692f
#define INV_PI        0.31830988618379067154f
#define INV_TWOPI     0.15915494309189533577f

struct TestClass_Generated_UBO_Data
{
  mat4 m_worldViewProjInv; 
  int m_height; 
  int m_width; 
  uint m_randomGens_capacity; 
  uint m_randomGens_size; 
  uint spheresMaterials_capacity; 
  uint spheresMaterials_size; 
  uint spheresPosRadius_capacity; 
  uint spheresPosRadius_size; 
  uint dummy_last;
};

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

mat4 make_float4x4_from_cols(vec4 a, vec4 b, vec4 c, vec4 d) { return mat4(a, b, c, d); } // swaped rows/cols logic between LiteMath <=> GLSL
mat4 make_float4x4_from_rows(vec4 a, vec4 b, vec4 c, vec4 d)                              // swaped rows/cols logic between LiteMath <=> GLSL
{
  return mat4(
      a.x, b.x, c.x, d.x,
      a.y, b.y, c.y, d.y,
      a.z, b.z, c.z, d.z,
      a.w, b.w, c.w, d.w
  );
}

mat4 translate4x4(vec3 delta)
{
  return mat4(vec4(1.0, 0.0, 0.0, 0.0),
              vec4(0.0, 1.0, 0.0, 0.0),
              vec4(0.0, 0.0, 1.0, 0.0),
              vec4(delta, 1.0));
}

mat4 rotate4x4X(float phi)
{
  return mat4(vec4(1.0f, 0.0f,  0.0f,           0.0f),
              vec4(0.0f, +cos(phi),  +sin(phi), 0.0f),
              vec4(0.0f, -sin(phi),  +cos(phi), 0.0f),
              vec4(0.0f, 0.0f,       0.0f,      1.0f));
}

mat4 rotate4x4Y(float phi)
{
  return mat4(vec4(+cos(phi), 0.0f, -sin(phi), 0.0f),
              vec4(0.0f,      1.0f, 0.0f,      0.0f),
              vec4(+sin(phi), 0.0f, +cos(phi), 0.0f),
              vec4(0.0f,      0.0f, 0.0f,      1.0f));
}

mat4 rotate4x4Z(float phi)
{
  return mat4(vec4(+cos(phi), sin(phi), 0.0f, 0.0f),
              vec4(-sin(phi), cos(phi), 0.0f, 0.0f),
              vec4(0.0f,      0.0f,     1.0f, 0.0f),
              vec4(0.0f,      0.0f,     0.0f, 1.0f));
}

mat4 inverse4x4(mat4 m) { return inverse(m); }
vec3 mul4x3(mat4 m, vec3 v) { return (m*vec4(v, 1.0f)).xyz; }
vec3 mul3x3(mat4 m, vec3 v) { return (m*vec4(v, 0.0f)).xyz; }

mat3 make_float3x3(vec3 a, vec3 b, vec3 c) { // different way than mat3(a,b,c)
  return mat3(a.x, b.x, c.x,
              a.y, b.y, c.y,
              a.z, b.z, c.z);
}

void set_row2f(inout mat2 m, int col, vec2 value) { m[0][col] = value.x; m[1][col] = value.y; }
void set_row3f(inout mat3 m, int col, vec3 value) { m[0][col] = value.x; m[1][col] = value.y; m[2][col] = value.z; }
void set_row4f(inout mat4 m, int col, vec4 value) { m[0][col] = value.x; m[1][col] = value.y; m[2][col] = value.z; m[3][col] = value.w; }

vec4 cross3(vec4 a, vec4 b) { return vec4(cross(a.xyz, b.xyz), 1.0f); }

struct Box4f 
{ 
  vec4 boxMin; 
  vec4 boxMax;
};  

#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8
#define KGEN_REDUCTION_LAST_STEP    16
#define RTC_RANDOM 
#define TEST_CLASS_H 
#define BASIC_PROJ_LOGIC_H 
#define MAXFLOAT FLT_MAX

uint NextState(inout RandomGen gen) {
  const uint x = (gen.state).x * 17 + (gen.state).y * 13123;
  (gen.state).x = (x << 13) ^ x;
  (gen.state).y ^= (x << 7);
  return x;
}

float epsilonOfPos(vec3 hitPos) { return max(max(abs(hitPos.x), max(abs(hitPos.y), abs(hitPos.z))), 2.0f*GEPSILON)*GEPSILON; }

void CoordinateSystem(vec3 v1, inout vec3 v2, inout vec3 v3) {
  float invLen = 1.0f;

  if (abs(v1.x) > abs(v1.y))
  {
    invLen = 1.0f / sqrt(v1.x*v1.x + v1.z*v1.z);
    (v2)  = vec3((-1.0f) * v1.z * invLen, 0.0f, v1.x * invLen);
  }
  else
  {
    invLen = 1.0f / sqrt(v1.y * v1.y + v1.z * v1.z);
    (v2)  = vec3(0.0f, v1.z * invLen, (-1.0f) * v1.y * invLen);
  }

  (v3) = cross(v1, (v2));
}

vec3 OffsRayPos(const vec3 a_hitPos, const vec3 a_surfaceNorm, const vec3 a_sampleDir) {
  const float signOfNormal2 = dot(a_sampleDir, a_surfaceNorm) < 0.0f ? -1.0f : 1.0f;
  const float offsetEps     = epsilonOfPos(a_hitPos);
  return a_hitPos + signOfNormal2*offsetEps*a_surfaceNorm;
}

vec3 MapSampleToCosineDistribution(float r1, float r2, vec3 direction, vec3 hit_norm, float power) {
  if(power >= 1e6f)
    return direction;

  const float sin_phi = sin(M_TWOPI * r1);
  const float cos_phi = cos(M_TWOPI * r1);

  //sincos(2.0f*r1*3.141592654f, &sin_phi, &cos_phi);

  const float cos_theta = pow(1.0f - r2, 1.0f / (power + 1.0f));
  const float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

  vec3 deviation;
  deviation.x = sin_theta*cos_phi;
  deviation.y = sin_theta*sin_phi;
  deviation.z = cos_theta;

  vec3 ny = direction,  nx,  nz;
  CoordinateSystem(ny, nx, nz);

  {
    vec3 temp = ny;
    ny = nz;
    nz = temp;
  }

  vec3 res = nx*deviation.x + ny*deviation.y + nz*deviation.z;

  float invSign = dot(direction, hit_norm) > 0.0f ? 1.0f : -1.0f;

  if (invSign*dot(res, hit_norm) < 0.0f) // reflected ray is below surface #CHECK_THIS
  {
    res = (-1.0f)*nx*deviation.x + ny*deviation.y - nz*deviation.z;
    //belowSurface = true;
  }

  return res;
}

vec2 rndFloat2_Pseudo(inout RandomGen gen) {
  uint x = NextState(gen); 

  const uint x1 = (x * (x * x * 15731 + 74323) + 871483);
  const uint y1 = (x * (x * x * 13734 + 37828) + 234234);

  const float scale     = (1.0f / 4294967296.0f);

  return vec2(float((x1)), float((y1)))*scale;
}

vec3 GetMtlDiffuseColor(in SphereMaterial a_mtl) { return a_mtl.color.xyz; }

vec3 GetMtlEmissiveColor(in SphereMaterial a_mtl) { return a_mtl.color.xyz*a_mtl.color.w; }

bool IsMtlEmissive(in SphereMaterial a_mtl) { return (a_mtl.color.w != 0.0f); }

vec3 EyeRayDir(float x, float y, float w, float h, mat4 a_mViewProjInv) {
  vec4 pos = vec4(2.0f * (x + 0.5f) / w - 1.0f, -2.0f * (y + 0.5f) / h + 1.0f, 0.0f, 1.0f);

  pos = (a_mViewProjInv*pos);
  pos /= pos.w;

  pos.y *= (-1.0f);

  return normalize(pos.xyz);
}

uint RealColorToUint32_f3(vec3 real_color) {
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  uint red = uint(r), green = uint(g), blue = uint(b);
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}

vec2 RaySphereHit(vec3 orig, vec3 dir, vec4 sphere) {
  const vec3 center = sphere.xyz;
  const float  radius = sphere.w;

  // Hearn and Baker equation 10-72 for when radius^2 << distance between origin and center
	// Also at https://www.cg.tuwien.ac.at/courses/EinfVisComp/Slides/SS16/EVC-11%20Ray-Tracing%20Slides.pdf
	// Assumes ray direction is normalized
	//dir = normalize(dir);
	const vec3 deltap = center - orig;
	const float ddp       = dot(dir, deltap);
	const float deltapdot = dot(deltap, deltap);

	// old way, "standard", though it seems to be worse than the methods above
	//float discriminant = ddp * ddp - deltapdot + radius * radius;
	vec3 remedyTerm = deltap - ddp * dir;
	float discriminant = radius * radius - dot(remedyTerm, remedyTerm);

  vec2 result = vec2(0,0);
	if (discriminant >= 0.0f)
	{
		const float sqrtVal = sqrt(discriminant);
		// include Press, William H., Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery, 
		// "Numerical Recipes in C," Cambridge University Press, 1992.
		const float q = (ddp >= 0) ? (ddp + sqrtVal) : (ddp - sqrtVal);
		// we don't bother testing for division by zero
		const float t1 = q;
		const float t2 = (deltapdot - radius * radius) / q;
    result.x = min(t1, t2);
    result.y = max(t1, t2);
  }
  
  return result;
}

uint fakeOffset(uint x, uint y, uint pitch) { return y*pitch + x; }  // RTV pattern, for 2D threading


