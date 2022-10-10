#include <cmath>
#include <limits>
#include <cstdlib>
#include "our_gl.h"
#include <algorithm>
#define PI 3.1415926
Matrix ModelMatrix;
Matrix ViewMatrix;
Matrix ModelView;
Matrix Viewport;
Matrix Projection;

IShader::~IShader() {}

//视角矩阵
void viewport(int x, int y, int w, int h) {
    Viewport = Matrix::identity();
    Viewport[0][3] = x+w/2.f;
    Viewport[1][3] = y+h/2.f;
    Viewport[0][0] = w/2.f;
    Viewport[1][1] = h/2.f;
}

//投影矩阵
Matrix orthogonal(float l, float r, float b, float t, float n, float f){
    Matrix Orth = Matrix::identity();
    Orth[0][0] = 2. / (r - l);
	Orth[1][1] = 2. / (t - b);
	Orth[2][2] = 2. / (n - f);
	Orth[3][3] = 1.;
    Matrix pann = Matrix::identity();
    pann[0][3] = -(r + l) / 2.;
    pann[1][3] = -(t + b) / 2.;
    pann[2][3] = -(n + f) / 2.;
    
    return Orth * pann;
}

void projection(float coeff) {
    Projection = Matrix::identity();
    Projection[3][2] = coeff;
}

void projection(float l, float r, float n, float f, float t, float b) {
    Projection = Matrix::identity();
    Projection[0][0] = 2*n / (r - l);  Projection[0][1] = 0;                Projection[0][2] = (l+r)/(l-r);         Projection[0][3] = 0;
	Projection[1][0] = 0;	           Projection[1][1] = 2*n / (t - b);    Projection[1][2] = (b+t)/(b-t);         Projection[1][3] = 0;
	Projection[2][0] = 0;              Projection[2][1] = 0;                Projection[2][2] = (n+f) / (n - f);     Projection[2][3] = 2*n*f / (f - n);
	Projection[3][0] = 0;		       Projection[3][1] = 0;                Projection[3][2] = 1;	                Projection[3][3] = 0;
}

void projection(float fov, float aspect, float n, float f){
    float z_range = f - n;
	Projection = Matrix::identity();
	assert(fov > 0 && aspect > 0);
	assert(n > 0 && f > 0 && z_range > 0);
	Projection[1][1] = 1 / (float)tan(fov / 2);
	Projection[0][0] = Projection[1][1] / aspect;
	Projection[2][2] = -(n + f) / z_range;
	Projection[2][3] = -2 * n * f / z_range;
	Projection[3][2] = -1;
	Projection[3][3] = 0;
}

//变换矩阵
void get_view_matrix(Vec3f eye_pos){
    ViewMatrix = Matrix::identity();
    Matrix translate = Matrix::identity();
    translate[0][3] = -eye_pos[0];
    translate[1][3] = -eye_pos[1];
    translate[2][3] = -eye_pos[2];
    ViewMatrix = translate*ViewMatrix;
}

void get_model_matrix(float angle){
    Matrix rotation = Matrix::identity();
    angle = angle * PI / 180.f;
    rotation[0][0] = cos(angle);
    rotation[0][2] = -sin(angle);
    rotation[2][0] = sin(angle);
    rotation[2][2] = cos(angle);
    Matrix scale = Matrix::identity();
    scale[0][0] = 2.5;
    scale[1][1] = 2.5;
    scale[2][2] = 2.5;
    Matrix translate = Matrix::identity();
    ModelMatrix = translate * rotation * scale;
}

//计算质心坐标
Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2f P) {
    Vec3f s[2];
    for (int i=2; i--; ) {
        s[i][0] = C[i]-A[i];
        s[i][1] = B[i]-A[i];
        s[i][2] = A[i]-P[i];
    }
    Vec3f u = cross(s[0], s[1]);
    if (std::abs(u[2])>1e-2) 
        return Vec3f(1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z);
    return Vec3f(-1,1,1);
}

//绘制三角形
void triangle(Vec4f *pts, IShader &shader, TGAImage &image, float *zbuffer) {
    //初始化三角形边界框
    std::vector<float> x_arry{ pts[0][0], pts[1][0], pts[2][0] };
	std::vector<float> y_arry{ pts[0][1], pts[1][1], pts[2][1] };
	std::sort(x_arry.begin(), x_arry.end());
	std::sort(y_arry.begin(), y_arry.end());

    Vec2f bboxmin(floor(x_arry[0]),floor(y_arry[0]));
    Vec2f bboxmax(ceil(x_arry[2]),ceil(y_arry[2]));
    //当前像素坐标P，颜色color
    Vec2i P;
    TGAColor color;
    //遍历边界框中的每一个像素
    for (P.x=bboxmin.x; P.x<=bboxmax.x; P.x++) {
        for (P.y=bboxmin.y; P.y<=bboxmax.y; P.y++) {
            //c为当前P对应的质心坐标
            //这里pts除以了最后一个分量，实现了透视中的缩放，所以用于判断P是否在三角形内
            Vec3f c = barycentric(proj<2>(pts[0]/pts[0][3]), proj<2>(pts[1]/pts[1][3]), proj<2>(pts[2]/pts[2][3]), proj<2>(P));
            //插值计算P的zbuffer
            
            float z_P = (pts[0][2]/ pts[0][3])*c.x + (pts[0][2] / pts[1][3]) *c.y + (pts[0][2] / pts[2][3]) *c.z;
            z_P = z_P/(c.x + c.y + c.z);
            //P的任一质心分量小于0或者zbuffer小于已有zbuffer，不渲染
            if (c.x<0 || c.y<0 || c.z<0 || P.y<0 || P.x<0 || zbuffer[P.x+700*P.y]<z_P) continue;
            //调用片元着色器计算当前像素颜色
            bool discard = shader.fragment(c, color);
            if (!discard) {
                //zbuffer
                zbuffer[P.x+700*P.y] = z_P;
                //zbuffer.set(P.x, P.y, TGAColor(frag_depth));
                //为像素设置颜色
                image.set(P.x, P.y, color);
            }
        }
    }
}

void triangle(Vec4f *pts, IShader &shader, TGAImage &image, TGAImage &zbuffer) {
    //初始化三角形边界框
    Vec2f bboxmin( std::numeric_limits<float>::max(),  std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            //这里pts除以了最后一个分量，实现了透视中的缩放，所以作为边界框
            bboxmin[j] = std::min(bboxmin[j], pts[i][j]/pts[i][3]);
            bboxmax[j] = std::max(bboxmax[j], pts[i][j]/pts[i][3]);
        }
    }
    //当前像素坐标P，颜色color
    Vec2i P;
    TGAColor color;
    //遍历边界框中的每一个像素
    for (P.x=bboxmin.x; P.x<=bboxmax.x; P.x++) {
        for (P.y=bboxmin.y; P.y<=bboxmax.y; P.y++) {
            //c为当前P对应的质心坐标
            //这里pts除以了最后一个分量，实现了透视中的缩放，所以用于判断P是否在三角形内
            Vec3f c = barycentric(proj<2>(pts[0]/pts[0][3]), proj<2>(pts[1]/pts[1][3]), proj<2>(pts[2]/pts[2][3]), proj<2>(P));
            //插值计算P的zbuffer
            //pts[i]为三角形的三个顶点
            //pts[i][2]为三角形的z信息(0~255)
            //pts[i][3]为三角形的投影系数(1-z/c)
            
            float z_P = (pts[0][2]/ pts[0][3])*c.x + (pts[0][2] / pts[1][3]) *c.y + (pts[0][2] / pts[2][3]) *c.z;
            int frag_depth = std::max(0, std::min(255, int(z_P+.5)));
            //P的任一质心分量小于0或者zbuffer小于已有zbuffer，不渲染
            if (c.x<0 || c.y<0 || c.z<0 || zbuffer.get(P.x,P.y)[0]>frag_depth) continue;
            //调用片元着色器计算当前像素颜色
            bool discard = shader.fragment(c, color);
            if (!discard) {
                //zbuffer
                zbuffer.set(P.x, P.y, TGAColor(frag_depth));
                //为像素设置颜色
                image.set(P.x, P.y, color);
            }
        }
    }
}

