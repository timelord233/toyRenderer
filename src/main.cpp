#include <vector>
#include <iostream>
#include <algorithm>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

Model *model     = NULL;
float* zbuffer = NULL;
const int width  = 700;
const int height = 700;

const float f1 = (50 - 0.1) / 2.0;
const float f2 = (50 + 0.1) / 2.0;

Vec3f light_dir(0,1,1);
Vec3f       eye(0,0,6);
Vec3f    center(0,0,0);
Vec3f        up(0,1,0);

//高洛德着色器
struct GouraudShader : public IShader {
    //顶点着色器会将数据写入varying_intensity
    //片元着色器从varying_intensity中读取数据
    Vec3f varying_intensity; 
    mat<2, 3, float> varying_uv;
    //接受两个变量，(面序号，顶点序号)
    virtual Vec4f vertex(int iface, int nthvert) {
        //根据面序号和顶点序号读取模型对应顶点，并扩展为4维 
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        //变换顶点坐标到屏幕坐标（视角矩阵*投影矩阵*变换矩阵*v）
        mat<4, 4, float> uniform_M = Projection * ViewMatrix * ModelMatrix;
        mat<4, 4, float> uniform_MIT = (ViewMatrix * ModelMatrix).invert_transpose();
        gl_Vertex = uniform_M *gl_Vertex;
        gl_Vertex[0] = gl_Vertex[0]/gl_Vertex[3];
        gl_Vertex[1] = gl_Vertex[1]/gl_Vertex[3];
        gl_Vertex[2] = gl_Vertex[2]/gl_Vertex[3];
        gl_Vertex[3] = 1;
        gl_Vertex[0] = 0.5*width*(gl_Vertex[0]+1.0);
        gl_Vertex[1] = 0.5*height*(gl_Vertex[1]+1.0);
        gl_Vertex[2] = gl_Vertex[2] * f1 + f2;
        //计算光照强度（顶点法向量*光照方向）
        Vec3f normal = proj<3>(embed<4>(model->normal(iface, nthvert))).normalize();
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) *light_dir); // get diffuse lighting intensity
        return gl_Vertex;
    }
    //根据传入的质心坐标，颜色，以及varying_intensity计算出当前像素的颜色
    virtual bool fragment(Vec3f bar, TGAColor &color) {
        Vec2f uv = varying_uv * bar;
        TGAColor c = model->diffuse(uv);
        float intensity = varying_intensity*bar;
        color = c*intensity; 
        return false;                              
    }
};

//将一定阈值内的光照强度给替换为一种
struct ToonShader : public IShader {
    mat<3, 3, float> varying_tri;
    Vec3f          varying_ity;

    virtual ~ToonShader() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        gl_Vertex = Projection * ModelView * gl_Vertex;
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));

        varying_ity[nthvert] = model->normal(iface, nthvert) * light_dir;

        gl_Vertex = Viewport * gl_Vertex;
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) {
        float intensity = varying_ity * bar;
        if (intensity > .85) intensity = 1;
        else if (intensity > .60) intensity = .80;
        else if (intensity > .45) intensity = .60;
        else if (intensity > .30) intensity = .45;
        else if (intensity > .15) intensity = .30;
        color = TGAColor(255, 155, 0) * intensity;
        return false;
    }
};

//不对法向量进行插值，法向量来源于三角形边的叉积
struct FlatShader : public IShader {
    //三个点的信息
    mat<3, 3, float> varying_tri;

    virtual ~FlatShader() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        gl_Vertex = Projection * ModelView * gl_Vertex;
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        gl_Vertex = Viewport * gl_Vertex;
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) {

        Vec3f n = cross(varying_tri.col(1) - varying_tri.col(0), varying_tri.col(2) - varying_tri.col(0)).normalize();
        float intensity = n * light_dir;
        color = TGAColor(255, 255, 255) * intensity;
        return false;
    }
};

//Phong氏着色
struct PhongShader : public IShader {
    mat<2, 3, float> varying_uv;  // same as above
    mat<4, 4, float> uniform_M = Projection * ViewMatrix * ModelMatrix;
    mat<4, 4, float> uniform_MIT = (ViewMatrix * ModelMatrix).invert_transpose();
    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        gl_Vertex = Projection * ViewMatrix * ModelMatrix * gl_Vertex;
        gl_Vertex[0] = gl_Vertex[0]/gl_Vertex[3];
        gl_Vertex[1] = gl_Vertex[1]/gl_Vertex[3];
        gl_Vertex[2] = gl_Vertex[2]/gl_Vertex[3];
        gl_Vertex[3] = 1;
        gl_Vertex[0] = 0.5*width*(gl_Vertex[0]+1.0);
        gl_Vertex[1] = 0.5*height*(gl_Vertex[1]+1.0);
        gl_Vertex[2] = gl_Vertex[2] * f1 + f2;
        return gl_Vertex; // transform it to screen coordinates
    }
    virtual bool fragment(Vec3f bar, TGAColor& color) {
        Vec2f uv = varying_uv * bar;
        Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        Vec3f r = (n * (n * l * 2.f) - l).normalize();   // reflected light
        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n * l);
        TGAColor c = model->diffuse(uv);
        color = c;
        for (int i = 0; i < 3; i++) color[i] = std::min<float>(5 + c[i] * (diff + .6 * spec), 255);
        return false;
    }
};

//计算质心坐标
Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
    Vec3f s[2];
    //计算[AB,AC,PA]的x和y分量
    for (int i=2; i--; ) {
        s[i][0] = C[i]-A[i];
        s[i][1] = B[i]-A[i];
        s[i][2] = A[i]-P[i];
    }
    //[u,v,1]和[AB,AC,PA]对应的x和y向量都垂直，所以叉乘
    Vec3f u = cross(s[0], s[1]);
    //三点共线时，会导致u[2]为0，此时返回(-1,1,1)
    if (std::abs(u[2])>1e-2)
        //若1-u-v，u，v全为大于0的数，表示点在三角形内部
        return Vec3f(1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z);
    return Vec3f(-1,1,1);
}

void triangle1(Vec3f *pts, float *zbuffer, TGAImage &image, TGAColor color) {
    std::vector<float> x_arry{ pts[0][0], pts[1][0], pts[2][0] };
	std::vector<float> y_arry{ pts[0][1], pts[1][1], pts[2][1] };
	std::sort(x_arry.begin(), x_arry.end());
	std::sort(y_arry.begin(), y_arry.end());

    Vec2f bboxmin(floor(x_arry[0]),floor(y_arry[0]));
    Vec2f bboxmax(ceil(x_arry[2]),ceil(y_arry[2]));

    Vec3i P;
    //遍历边框中的每一个点
    for (P.x=bboxmin.x; P.x<=bboxmax.x; P.x++) {
        for (P.y=bboxmin.y; P.y<=bboxmax.y; P.y++) {
            Vec3f bc_screen  = barycentric(pts[0], pts[1], pts[2], P);
            float w_reciprocal = 1.0 / (bc_screen.x + bc_screen.y + bc_screen.z);
			float z_interpolated = bc_screen.x * pts[0][2] + bc_screen.y * pts[1][2] + bc_screen.z * pts[2][2];
			z_interpolated *= w_reciprocal;
            //质心坐标有一个负值，说明点在三角形外
            if (bc_screen.x<0 || bc_screen.y<0 || bc_screen.z<0) continue;
            if (zbuffer[int(P.x+P.y*width)] > z_interpolated) {
                zbuffer[int(P.x+P.y*width)] = z_interpolated;
                image.set(P.x, P.y, color);
            }
        }
    }
}

Vec3f world2screen(Vec3f v) {
    return Vec3f(int((v.x+1.)*width/2.+.5), int((v.y+1.)*height/2.+.5), v.z);
}

int main(int argc, char** argv) {
    //加载模型
    if (2==argc) {
        model = new Model(argv[1]);
    } else {
        model = new Model("../obj/african_head/african_head.obj");
    }
    //初始化变换矩阵，投影矩阵，视角矩阵
    get_model_matrix(45.0f);
    get_view_matrix(eye);
    //projection(-1.f/(eye-center).norm());
    projection(45.f, 1. * width / height, .1f, 50.f);
    viewport(0, 0, width, height);
    light_dir.normalize();
    //初始化image和zbuffer
    TGAImage image  (width, height, TGAImage::RGB);
    zbuffer = new float[width*height];
    for (int i=0; i<width*height; i++) {
        zbuffer[i] = std::numeric_limits<float>::infinity();
    }
    TGAImage zbuffer1(width, height, TGAImage::GRAYSCALE);
    //实例化高洛德着色
    //GouraudShader shader;
    //实例化Phong着色
    PhongShader shader;
    //实例化Toon着色
	//ToonShader shader;


    for (int i=0; i<model->nfaces(); i++) {
        //屏幕坐标，世界坐标
        Vec3f screen_coords[3];
        Vec3f world_coords[3];
        std::vector<int> face = model->face(i);
        for (int j = 0; j < 3; j++) {
            Vec3f v = model->vert(face[j]);
            //世界坐标转屏幕坐标
            Vec4f temp =  Projection  * ViewMatrix * ModelMatrix* embed<4>(model->vert(face[j]));
            //screen_coords[j] = world2screen(model->vert(face[j]));
            screen_coords[j][0] = temp[0]/temp[3];
            screen_coords[j][1] = temp[1]/temp[3];
            screen_coords[j][2] = temp[2]/temp[3];
            screen_coords[j][0] = 0.5*width*(screen_coords[j][0]+1.0);
            screen_coords[j][1] = 0.5*height*(screen_coords[j][1]+1.0);
            screen_coords[j][2] = screen_coords[j][2]*f1 + f2;
            world_coords[j] = v;
        }
        //世界坐标用于计算法向量
        Vec3f n = cross((world_coords[2] - world_coords[0]),(world_coords[1] - world_coords[0]));
        n.normalize();
        float intensity = -(n * light_dir);
        //背面裁剪
        if (intensity > 0) {
            triangle1(screen_coords, zbuffer, image, TGAColor(intensity*255, intensity*255, intensity*255, 255));
        }
    }


    for (int i=0; i<model->nfaces(); i++) {
        Vec4f screen_coords[3];
        for (int j=0; j<3; j++) {
            //通过顶点着色器读取模型顶点
            //变换顶点坐标到屏幕坐标（视角矩阵*投影矩阵*变换矩阵*v） ***其实并不是真正的屏幕坐标，因为没有除以最后一个分量
            //计算光照强度
            screen_coords[j] = shader.vertex(i, j);
        }
        //遍历完3个顶点，一个三角形光栅化完成
        //绘制三角形，triangle内部通过片元着色器对三角形着色
        //triangle(screen_coords, shader, image, zbuffer);
    }


    image.  flip_vertically();
    //zbuffer.flip_vertically();
    image.  write_tga_file("output.tga");
    //zbuffer.write_tga_file("zbuffer.tga");

    delete model;
    delete [] zbuffer;
    return 0;
}
