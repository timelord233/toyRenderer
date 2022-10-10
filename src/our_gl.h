#ifndef __OUR_GL_H__
#define __OUR_GL_H__

#include "tgaimage.h"
#include "geometry.h"

extern Matrix ModelMatrix;
extern Matrix ViewMatrix;
extern Matrix ModelView;
extern Matrix Viewport;
extern Matrix Projection;

void viewport(int x, int y, int w, int h);
void projection(float l, float r, float n, float f, float t, float b); 
void projection(float coeff);
void projection(float fov, float aspect, float n, float f);
void lookat(Vec3f eye, Vec3f center, Vec3f up);
void get_view_matrix(Vec3f eye_pos);
void get_model_matrix(float angle);

struct IShader {
    virtual ~IShader();
    virtual Vec4f vertex(int iface, int nthvert) = 0;
    virtual bool fragment(Vec3f bar, TGAColor &color) = 0;
};

void triangle(Vec4f *pts, IShader &shader, TGAImage &image, float *zbuffer);
void triangle(Vec4f *pts, IShader &shader, TGAImage &image, TGAImage &zbuffer);

#endif //__OUR_GL_H__

