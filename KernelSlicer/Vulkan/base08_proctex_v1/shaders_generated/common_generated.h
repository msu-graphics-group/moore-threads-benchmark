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

const int TOTAL_IMPLEMANTATIONS = 4;
const int BRANCHING_LITE = 0;
const int BRANCHING_MEDIUM = 1;
const int BRANCHING_HEAVY = 2;
const int AA = 2;
const int NUMBER_OF_POINTS = 8;

#define M_PI          3.14159265358979323846f
#define M_TWOPI       6.28318530717958647692f
#define INV_PI        0.31830988618379067154f
#define INV_TWOPI     0.15915494309189533577f

struct ProcRender2D_Generated_UBO_Data
{
  float m_time; 
  int m_usedImplementations; 
  uint allProcTextures_capacity; 
  uint allProcTextures_size; 
  uint all_references_capacity; 
  uint all_references_size; 
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
#define MAXFLOAT FLT_MAX

vec4 floor4(vec4 v) {
  return vec4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

vec4 mod289f4(vec4 x) {
  return x - floor4(x * (1.0 / 289.0)) * 289.0;
}

vec3 floor3(vec3 v) {
  return vec3(floor(v.x), floor(v.y), floor(v.z));
}

vec2 fract2(vec2 v) {
  return v - floor(v);
}

vec2 mix2(vec2 x, vec2 y, float a) {
  return vec2(mix(x.x, y.x, a), mix(x.y, y.y, a));
}

vec4 step4_(vec4 edge, vec4 x) {
  return vec4(x.x < edge.x ? 0.0f : 1.0f, x.y < edge.y ? 0.0f : 1.0f, x.z < edge.z ? 0.0f : 1.0f, x.w < edge.w ? 0.0f : 1.0f);
}

vec3 mod289f3(vec3 x) {
  return x - floor3(x * (1.0 / 289.0)) * 289.0;
}

vec2 hash2(vec2 p) {
  const vec2 p1 = vec2(dot(p, vec2(123.4f, 748.6f)),dot(p, vec2(547.3f,659.3f)));
  return fract2(vec2(sin(p1.x)*5232.85324f,sin(p1.y)*5232.85324f));   
}

vec4 permute(vec4 x) {
  return mod289f4(((x*34.0) + 1.0)*x);
}

vec3 fract3(vec3 v) {
  return v - floor3(v);
}

vec4 mix4(vec4 x, vec4 y, float a) {
  return vec4(mix(x.x, y.x, a), mix(x.y, y.y, a), mix(x.z, y.z, a), mix(x.w, y.w, a));
}

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  const vec3 tmp = 2.0f*3.141592653f*(c*t+d);
	return a + b*vec3(cos(tmp.x),cos(tmp.y),cos(tmp.z));
}

vec2 cdiv(vec2 a, vec2 b) {
	return vec2(((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y)),((a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y)));
}

float circ(vec2 pos, vec2 c, float s) {
  c = abs(pos - c);
  c = min(c, 1.0f - c);

  return smoothstep(0.0f, 0.002f, sqrt(s) - sqrt(dot(c, c))) * -1.0f;
}

mat4 rotate(float theta) {
	float s = sin(theta);
	float c = cos(theta);
	return mat4(c,-s,0.0f,0.0f,s,c,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f);
}

vec4 taylorInvSqrt(vec4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade(vec3 t) {
  return t*t*t*(t*(t*6.0 - 15.0) + 10.0);
}

vec4 fract4(vec4 v) {
  return v - floor4(v);
}

vec2 cpow2(vec2 c) {
	return vec2(c.x * c.x - c.y * c.y,2.0f * c.x * c.y);
}

vec4 step4(float edge, vec4 x) {
  return vec4(x.x < edge ? 0.0f : 1.0f, x.y < edge ? 0.0f : 1.0f, x.z < edge ? 0.0f : 1.0f, x.w < edge ? 0.0f : 1.0f);
}

vec4 abs4(vec4 a) {
  return vec4(abs(a.x), abs(a.y), abs(a.z), abs(a.w));
}

vec2 cmul(vec2 a, vec2 b) {
	return vec2(a.x * b.x - a.y * b.y,a.x * b.y + b.x * a.y);
}

vec2 grad(ivec2 z) {
  // 2D to 1D  (feel free to replace by some other)
  int n = z.x+z.y*11111;
  // Hugo Elias hash (feel free to replace by another one)
  n = (n<<13)^n;
  n = (n*(n*n*15731+789221)+1376312589)>>16;
  // Perlin style vectors
  n &= 7;
  vec2 gr = vec2(n&1,n>>1)*2.0-1.0;
  return ( n>=6 ) ? vec2(0.0f,gr.x) : 
         ( n>=4 ) ? vec2(gr.x,0.0f) : gr;                           
}

float reflection(vec2 c) {
  const float time = -10.0f;
  const float rotationDuration = 1.0f;
	vec2 z = vec2(0.0f,0.0f); // z0
	vec2 dc = vec2(0.0f,0.0f); // Derivate of c

	const float h2 = 1.5; // Height of light
  const vec2 normal2 = normalize(vec2(-1.0f,1.0f));
  const vec4 normal4 = vec4(normal2.x,normal2.y,0.0f,0.0f);

	vec4 angle4 = rotate(time / rotationDuration) * normal4; // Light always from top left
  vec2 angle = vec2(angle4.x,angle4.y);

	for (int i = 0; i < 300; i++) {
		dc = 2.0f * cmul(dc, z) + vec2(1.0,0.0);
		z = cpow2(z) + c;

		if (length(z) > 100.0f) { // Outside lighting calculation formula
			vec2 slope = normalize(cdiv(z, dc));
			float reflection = dot(slope, angle) + h2;
			reflection = reflection / (1.0 + h2); // Lower value to max 1.0
			if (reflection < 0.0) {
				reflection = 0.0;
			}
			return reflection;
		}
	}

	return -1.0;
}

float mu(vec2 a) {return a.x*a.y;}

float waterlayer(vec2 tc) {
  vec2 uv = vec2(mod(tc.x, 1.0f),mod(tc.y, 1.0f));
  float ret = 1.0;
  ret += circ(uv, vec2(0.37378,0.277169), 0.0268181);
  ret += circ(uv, vec2(0.0317477,0.540372), 0.0193742);
  ret += circ(uv, vec2(0.430044,0.882218), 0.0232337);
  ret += circ(uv, vec2(0.641033,0.695106), 0.0117864);
  ret += circ(uv, vec2(0.0146398,0.0791346), 0.0299458);
  ret += circ(uv, vec2(0.43871,0.394445), 0.0289087);
  ret += circ(uv, vec2(0.909446,0.878141), 0.028466);
  ret += circ(uv, vec2(0.310149,0.686637), 0.0128496);
  ret += circ(uv, vec2(0.928617,0.195986), 0.0152041);
  ret += circ(uv, vec2(0.0438506,0.868153), 0.0268601);
  ret += circ(uv, vec2(0.308619,0.194937), 0.00806102);
  ret += circ(uv, vec2(0.349922,0.449714), 0.00928667);
  ret += circ(uv, vec2(0.0449556,0.953415), 0.023126);
  ret += circ(uv, vec2(0.117761,0.503309), 0.0151272);
  ret += circ(uv, vec2(0.563517,0.244991), 0.0292322);
  ret += circ(uv, vec2(0.566936,0.954457), 0.00981141);
  ret += circ(uv, vec2(0.0489944,0.200931), 0.0178746);
  ret += circ(uv, vec2(0.569297,0.624893), 0.0132408);
  ret += circ(uv, vec2(0.298347,0.710972), 0.0114426);
  ret += circ(uv, vec2(0.878141,0.771279), 0.00322719);
  ret += circ(uv, vec2(0.150995,0.376221), 0.00216157);
  ret += circ(uv, vec2(0.119673,0.541984), 0.0124621);
  ret += circ(uv, vec2(0.629598,0.295629), 0.0198736);
  ret += circ(uv, vec2(0.334357,0.266278), 0.0187145);
  ret += circ(uv, vec2(0.918044,0.968163), 0.0182928);
  ret += circ(uv, vec2(0.965445,0.505026), 0.006348);
  ret += circ(uv, vec2(0.514847,0.865444), 0.00623523);
  ret += circ(uv, vec2(0.710575,0.0415131), 0.00322689);
  ret += circ(uv, vec2(0.71403,0.576945), 0.0215641);
  ret += circ(uv, vec2(0.748873,0.413325), 0.0110795);
  ret += circ(uv, vec2(0.0623365,0.896713), 0.0236203);
  ret += circ(uv, vec2(0.980482,0.473849), 0.00573439);
  ret += circ(uv, vec2(0.647463,0.654349), 0.0188713);
  ret += circ(uv, vec2(0.651406,0.981297), 0.00710875);
  ret += circ(uv, vec2(0.428928,0.382426), 0.0298806);
  ret += circ(uv, vec2(0.811545,0.62568), 0.00265539);
  ret += circ(uv, vec2(0.400787,0.74162), 0.00486609);
  ret += circ(uv, vec2(0.331283,0.418536), 0.00598028);
  ret += circ(uv, vec2(0.894762,0.0657997), 0.00760375);
  ret += circ(uv, vec2(0.525104,0.572233), 0.0141796);
  ret += circ(uv, vec2(0.431526,0.911372), 0.0213234);
  ret += circ(uv, vec2(0.658212,0.910553), 0.000741023);
  ret += circ(uv, vec2(0.514523,0.243263), 0.0270685);
  ret += circ(uv, vec2(0.0249494,0.252872), 0.00876653);
  ret += circ(uv, vec2(0.502214,0.47269), 0.0234534);
  ret += circ(uv, vec2(0.693271,0.431469), 0.0246533);
  ret += circ(uv, vec2(0.415,0.884418), 0.0271696);
  ret += circ(uv, vec2(0.149073,0.41204), 0.00497198);
  ret += circ(uv, vec2(0.533816,0.897634), 0.00650833);
  ret += circ(uv, vec2(0.0409132,0.83406), 0.0191398);
  ret += circ(uv, vec2(0.638585,0.646019), 0.0206129);
  ret += circ(uv, vec2(0.660342,0.966541), 0.0053511);
  ret += circ(uv, vec2(0.513783,0.142233), 0.00471653);
  ret += circ(uv, vec2(0.124305,0.644263), 0.00116724);
  ret += circ(uv, vec2(0.99871,0.583864), 0.0107329);
  ret += circ(uv, vec2(0.894879,0.233289), 0.00667092);
  ret += circ(uv, vec2(0.246286,0.682766), 0.00411623);
  ret += circ(uv, vec2(0.0761895,0.16327), 0.0145935);
  ret += circ(uv, vec2(0.949386,0.802936), 0.0100873);
  ret += circ(uv, vec2(0.480122,0.196554), 0.0110185);
  ret += circ(uv, vec2(0.896854,0.803707), 0.013969);
  ret += circ(uv, vec2(0.292865,0.762973), 0.00566413);
  ret += circ(uv, vec2(0.0995585,0.117457), 0.00869407);
  ret += circ(uv, vec2(0.377713,0.00335442), 0.0063147);
  ret += circ(uv, vec2(0.506365,0.531118), 0.0144016);
  ret += circ(uv, vec2(0.408806,0.894771), 0.0243923);
  ret += circ(uv, vec2(0.143579,0.85138), 0.00418529);
  ret += circ(uv, vec2(0.0902811,0.181775), 0.0108896);
  ret += circ(uv, vec2(0.780695,0.394644), 0.00475475);
  ret += circ(uv, vec2(0.298036,0.625531), 0.00325285);
  ret += circ(uv, vec2(0.218423,0.714537), 0.00157212);
  ret += circ(uv, vec2(0.658836,0.159556), 0.00225897);
  ret += circ(uv, vec2(0.987324,0.146545), 0.0288391);
  ret += circ(uv, vec2(0.222646,0.251694), 0.00092276);
  ret += circ(uv, vec2(0.159826,0.528063), 0.00605293);
	return max(ret, 0.0f);
}

vec3 JuliaGetColorf(float iter, vec2 z) {
  vec3 color;
  color.x = float(mod(iter,10.f))/9.f;
  color.y = float(mod(iter/10.f,10.f))/9.f;
  color.z = .5f + .5f*sin(iter/10.f-log2(1.f+log2(1.f+dot(z,z))));
  return color;
}

int mandel(float c_re, float c_im, int count) {
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) 
  {
    if (z_re * z_re + z_im * z_im > 4.)
        break;
    float new_re = z_re*z_re - z_im*z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
      
  }
  return i;
}

float voronoi(vec2 p, float iTime) {
    vec2 n = floor(p);
    vec2 f = fract2(p);
    float md = 5.0f;
    vec2 m = vec2(0.0f,0.0f);
    for (int i = -1;i<=1;i++) {
        for (int j = -1;j<=1;j++) {
            vec2 g = vec2(i,j);
            vec2 o = hash2(n+g);
            o = 0.5f+0.5f*vec2(sin(iTime + 5.038f*o.x),sin(iTime + 5.038f*o.y));
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d<md) {
              md = d;
              m = n+g+o;
            }
        }
    }
    return md;
}

float potential(vec2 c) {
	vec2 z = vec2(0.0,0.0); // z0
	int iter = 0;

	for (iter = 0; iter < 300; ++iter) {
		z = cpow2(z) + c; // z_n+1 = z_n^2 + c
		float absZ = length(z); // |z|
		if (absZ > 100000.0f) {
			return abs(log(log2(absZ)) - (float(iter) + 1.0f) * log(2.0f));
		}
	}

	return -1.0;
}

float cnoise(vec3 P) {
  vec3 Pi0 = floor3(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0, 1.0, 1.0); // Integer part + 1
  Pi0 = mod289f3(Pi0);
  Pi1 = mod289f3(Pi1);
  vec3 Pf0 = fract3(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0, 1.0, 1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
  vec4 iz0 = vec4(Pi0.z, Pi0.z, Pi0.z, Pi0.z);
  vec4 iz1 = vec4(Pi1.z, Pi1.z, Pi1.z, Pi1.z);

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract4(floor4(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract4(gx0);
  vec4 gz0 = vec4(0.5, 0.5, 0.5, 0.5) - abs4(gx0) - abs4(gy0);
  vec4 sz0 = step4_(gz0, vec4(0.0, 0.0, 0.0, 0.0));
  gx0 -= sz0 * (step4(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step4(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract4(floor4(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract4(gx1);
  vec4 gz1 = vec4(0.5, 0.5, 0.5, 0.5) - abs4(gx1) - abs4(gy1);
  vec4 sz1 = step4_(gz1, vec4(0.0, 0.0, 0.0, 0.0));
  gx1 -= sz1 * (step4_(vec4(0.0f), gx1) - 0.5f);
  gy1 -= sz1 * (step4_(vec4(0.0f), gy1) - 0.5f);

  vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
  vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
  vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
  vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
  vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
  vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
  vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
  vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.y, Pf0.z));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.x, Pf1.y, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.x, Pf0.y, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.y, Pf1.z));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix4(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix2(vec2(n_z.x, n_z.y), vec2(n_z.z, n_z.w), fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

vec3 awesomePalette(float t) {
	return palette(t, vec3(0.5,0.5,0.5), vec3(0.5,0.5,0.5), vec3(1.0,1.0,1.0), vec3(0.0,0.1,0.2));
}

vec3 mandelbrot2DFunc(vec2 tc) {
  const int index = mandel((tc.x-0.5f)*1.25f, tc.y*1.25f, 100);

  const int r1 = min((index*128)/32, 255);
  const int g1 = min((index*128)/25, 255);
  const int b1 = min((index*index), 255);
  const float fr1 = float(r1)/255.0f;
  const float fg1 = float(g1)/255.0f;
  const float fb1 = float(b1)/255.0f;

  return vec3(fr1,fg1,fb1); 
}

vec3 water(vec2 uv, vec3 cdir) {
  const float iTime       = 0.5f;
  const vec3 WATER_COL = vec3(0.0f,0.4453f,0.7305f);
  const vec3 WATER2_COL = vec3(0.0f,0.4180f,0.6758f);
  const vec3 FOAM_COL = vec3(0.8125f,0.9609f,0.9648f);
  const float M_2PI = 6.283185307f;
  const float M_6PI = 18.84955592f;

  uv *= vec2(0.25);

  // Parallax height distortion with two directional waves at
  // slightly different angles.
  vec2 a = 0.025f * vec2(cdir.x,cdir.z) / cdir.y; // Parallax offset
  float h = sin(uv.x + iTime); // Height at UV
  uv += a * h;
  h = sin(0.841471f * uv.x - 0.540302f * uv.y + iTime);
  uv += a * h;
    
  // Texture distortion
  float d1 = mod(uv.x + uv.y, M_2PI);
  float d2 = mod((uv.x + uv.y + 0.25f) * 1.3f, M_6PI);
  d1 = iTime * 0.07f + d1;
  d2 = iTime * 0.5f + d2;
  vec2 dist = vec2(sin(d1) * 0.15f + sin(d2) * 0.05f,cos(d1) * 0.15f + cos(d2) * 0.05f);
    
  vec3 ret = mix(WATER_COL, WATER2_COL, waterlayer(uv + vec2(dist.x,dist.y)));
  ret = mix(ret, FOAM_COL, waterlayer(vec2(1.0f) - uv - vec2(dist.y,dist.x)));
  return ret;
}

float noise(vec2 p) {
  ivec2 i = ivec2(floor( p ));
  vec2 f = fract( p );

  vec2 u = f*f*(3.0f - 2.0f*f); // feel free to replace by a quintic smoothstep instead
  return mix( mix( dot( grad( i + ivec2(0,0) ), f - vec2(0.0,0.0) ), 
                   dot( grad( i + ivec2(1,0) ), f - vec2(1.0,0.0) ), u.x),
              mix( dot( grad( i + ivec2(0,1) ), f - vec2(0.0,1.0) ), 
                   dot( grad( i + ivec2(1,1) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

vec3 render_ps3dmandelbrot(vec2 tc) {
  const float time = -8.5f;
  const float centerDuration   = 31.0f;
  const float rotationDuration = 53.0f;
  const vec2 defaultCenter = vec2(-1.5f,-0.25f);
  const vec2 currentCenter = vec2(-1.0f,+0.50f);
  const vec3 insideColor = vec3(0.1f,0.12f,0.15f);
  const float currentZoom      = 1.25f;
    
  // Mix between base poistion and target position
	float mixFactor = 1.0f - (0.5f + 0.5f * cos(time / centerDuration * 2.0f * 3.141592653589793f));

	// Zoom and position calculation
	float zoom    = exp2(-currentZoom * mixFactor);
	float maxZoom = exp2(-currentZoom);
	
  vec4 tmp = rotate(time / rotationDuration) * vec4(tc.x * zoom,tc.y * zoom,0.0f,0.0f); 
  vec2 c = mix(currentCenter, defaultCenter, zoom / (1.0f - maxZoom) - maxZoom) + vec2(tmp.x,tmp.y);
	
  float pot = potential(c);
	float ref = reflection(c);
  float intensity = 0.7f * (fract(pot) * ref) + 0.3f;
   
  vec3 fragColor;
  vec3 color = awesomePalette(time / 50.0f + pot / 40.0f);
	if (ref < 0.0f) 
		fragColor = insideColor;
	else 
  { 
		fragColor = color * intensity + vec3(intensity) * 0.3f + clamp(ref - 0.5f, 0.0f, 1.0f) * pow((1.0f - fract(pot)), 30.0f);
		fragColor = clamp(fragColor, 0.0f, 1.0f);
	}

  return fragColor;
}

float ov(vec2 p, float iTime) {
    float v = 0.0f;
    float a = 0.4f;
    for (int i = 0;i<3;i++) {
        v+= voronoi(p, iTime)*a;
        p*=2.0f;
        a*=0.5f;
    }
    return v;
}

vec3 JuliaFunc(vec2 tc) {
  float fTime = 5.25f;
  vec2 uv = tc;
  
  uv-=.5f; 
  vec2 c = vec2(.7885f*cos(.5f*fTime),.7885f*sin(.5*fTime));
  vec3 color;
  vec2 z = uv;
  float iter=0.;
  float warp=2.;
  for(int i=0;i<500;++i){
      z=vec2(z.x*z.x-z.y*z.y,warp*z.x*z.y)+c;
      if (length(z)>5.0f) 
        break;
      iter++;
  }

  return clamp(JuliaGetColorf(iter,z), 0.0f, 1.0f);
}

float octave(vec3 pos, int octaves, float persistence) {
  float total = 0.0f;
  float frequency = 10.0f;
  float amplitude = 3.0f;
  float maxValue = 0.0f;

  for (int i = 0; i < octaves; i++)
  {
    total += cnoise(pos * frequency) * amplitude;

    maxValue += amplitude;

    amplitude *= persistence;
    frequency *= 2;
  }

  return total / maxValue;
}

float CheckerSignMuFract(vec2 u) { return sign(mu(.5-fract(u))); }

uint RealColorToUint32_f3(vec3 real_color) {
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  uint red = uint(r), green = uint(g), blue = uint(b);
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}


