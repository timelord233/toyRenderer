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

//�������ɫ��
struct GouraudShader : public IShader {
    //������ɫ���Ὣ����д��varying_intensity
    //ƬԪ��ɫ����varying_intensity�ж�ȡ����
    Vec3f varying_intensity; 
    mat<2, 3, float> varying_uv;
    //��������������(����ţ��������)
    virtual Vec4f vertex(int iface, int nthvert) {
        //��������źͶ�����Ŷ�ȡģ�Ͷ�Ӧ���㣬����չΪ4ά 
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        //�任�������굽��Ļ���꣨�ӽǾ���*ͶӰ����*�任����*v��
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
        //�������ǿ�ȣ����㷨����*���շ���
        Vec3f normal = proj<3>(embed<4>(model->normal(iface, nthvert))).normalize();
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) *light_dir); // get diffuse lighting intensity
        return gl_Vertex;
    }
    //���ݴ�����������꣬��ɫ���Լ�varying_intensity�������ǰ���ص���ɫ
    virtual bool fragment(Vec3f bar, TGAColor &color) {
        Vec2f uv = varying_uv * bar;
        TGAColor c = model->diffuse(uv);
        float intensity = varying_intensity*bar;
        color = c*intensity; 
        return false;                              
    }
};

//��һ����ֵ�ڵĹ���ǿ�ȸ��滻Ϊһ��
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

//���Է��������в�ֵ����������Դ�������αߵĲ��
struct FlatShader : public IShader {
    //���������Ϣ
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

//Phong����ɫ
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

//������������
Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
    Vec3f s[2];
    //����[AB,AC,PA]��x��y����
    for (int i=2; i--; ) {
        s[i][0] = C[i]-A[i];
        s[i][1] = B[i]-A[i];
        s[i][2] = A[i]-P[i];
    }
    //[u,v,1]��[AB,AC,PA]��Ӧ��x��y��������ֱ�����Բ��
    Vec3f u = cross(s[0], s[1]);
    //���㹲��ʱ���ᵼ��u[2]Ϊ0����ʱ����(-1,1,1)
    if (std::abs(u[2])>1e-2)
        //��1-u-v��u��vȫΪ����0��������ʾ�����������ڲ�
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
    //�����߿��е�ÿһ����
    for (P.x=bboxmin.x; P.x<=bboxmax.x; P.x++) {
        for (P.y=bboxmin.y; P.y<=bboxmax.y; P.y++) {
            Vec3f bc_screen  = barycentric(pts[0], pts[1], pts[2], P);
            float w_reciprocal = 1.0 / (bc_screen.x + bc_screen.y + bc_screen.z);
			float z_interpolated = bc_screen.x * pts[0][2] + bc_screen.y * pts[1][2] + bc_screen.z * pts[2][2];
			z_interpolated *= w_reciprocal;
            //����������һ����ֵ��˵��������������
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
    //����ģ��
    if (2==argc) {
        model = new Model(argv[1]);
    } else {
        model = new Model("../obj/african_head/african_head.obj");
    }
    //��ʼ���任����ͶӰ�����ӽǾ���
    get_model_matrix(45.0f);
    get_view_matrix(eye);
    //projection(-1.f/(eye-center).norm());
    projection(45.f, 1. * width / height, .1f, 50.f);
    viewport(0, 0, width, height);
    light_dir.normalize();
    //��ʼ��image��zbuffer
    TGAImage image  (width, height, TGAImage::RGB);
    zbuffer = new float[width*height];
    for (int i=0; i<width*height; i++) {
        zbuffer[i] = std::numeric_limits<float>::infinity();
    }
    TGAImage zbuffer1(width, height, TGAImage::GRAYSCALE);
    //ʵ�����������ɫ
    //GouraudShader shader;
    //ʵ����Phong��ɫ
    PhongShader shader;
    //ʵ����Toon��ɫ
	//ToonShader shader;


    for (int i=0; i<model->nfaces(); i++) {
        //��Ļ���꣬��������
        Vec3f screen_coords[3];
        Vec3f world_coords[3];
        std::vector<int> face = model->face(i);
        for (int j = 0; j < 3; j++) {
            Vec3f v = model->vert(face[j]);
            //��������ת��Ļ����
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
        //�����������ڼ��㷨����
        Vec3f n = cross((world_coords[2] - world_coords[0]),(world_coords[1] - world_coords[0]));
        n.normalize();
        float intensity = -(n * light_dir);
        //����ü�
        if (intensity > 0) {
            triangle1(screen_coords, zbuffer, image, TGAColor(intensity*255, intensity*255, intensity*255, 255));
        }
    }


    for (int i=0; i<model->nfaces(); i++) {
        Vec4f screen_coords[3];
        for (int j=0; j<3; j++) {
            //ͨ��������ɫ����ȡģ�Ͷ���
            //�任�������굽��Ļ���꣨�ӽǾ���*ͶӰ����*�任����*v�� ***��ʵ��������������Ļ���꣬��Ϊû�г������һ������
            //�������ǿ��
            screen_coords[j] = shader.vertex(i, j);
        }
        //������3�����㣬һ�������ι�դ�����
        //���������Σ�triangle�ڲ�ͨ��ƬԪ��ɫ������������ɫ
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
