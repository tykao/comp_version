#include <stdio.h>
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void rotation(Mat F,Mat &R,Mat &T,int r1);
Mat findFundamental(Mat k1, Mat k2, int N);
vector< DMatch > matchingFunc(  Mat descriptors_1, Mat descriptors_2);
Mat cameraMatrix();

void outputPly( String file, Mat img_1, vector<KeyPoint> m_LeftKey, Mat img_2, vector<KeyPoint> m_RightKey, vector<DMatch> m_Matches);

void findFundamentalMatrices(Mat img_1, Mat img_2)
{
    //  Task 1.1 Feature Detection.
    //  Using SIFT, SURF to detect the keypoint
    
    //set min Hessian
    int minHessian = 1400;
    
    Ptr<SURF> surfDetector = SURF::create();
    surfDetector->setHessianThreshold(minHessian);
    surfDetector->setExtended (true);
    
    std::vector<KeyPoint> surf_keypoints_1, surf_keypoints_2;
    Mat surf_descriptors_1, surf_descriptors_2;
    
    surfDetector->detectAndCompute( img_1, Mat(), surf_keypoints_1, surf_descriptors_1 );
    surfDetector->detectAndCompute( img_2, Mat(), surf_keypoints_2, surf_descriptors_2 );
    cout<<"Using SURF"<<endl;
    
    cout << "Image1 = "<< endl << " "  << surf_descriptors_1.size() << endl << endl;
    cout << "Image2 = "<< endl << " "  << surf_descriptors_2.size() << endl << endl;
    
    
    Ptr<SIFT> siftDetector = SIFT::create();
    
    std::vector<KeyPoint> sift_keypoints_1, sift_keypoints_2;
    Mat sift_descriptors_1, sift_descriptors_2;
    
    siftDetector->detectAndCompute( img_1, Mat(), sift_keypoints_1, sift_descriptors_1 );
    siftDetector->detectAndCompute( img_2, Mat(), sift_keypoints_2, sift_descriptors_2 );
    cout<<"Using SIFT"<<endl;
    
    cout << "point size of image1 = "<< endl << " "  << sift_descriptors_1.size() << endl << endl;
    cout << "point size of image2 = "<< endl << " "  << sift_descriptors_2.size() << endl << endl;
    
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    
    //------------------using SIFT--------------------------------------------------------
    //keypoints_1 = sift_keypoints_1, keypoints_2 = sift_keypoints_2;
    //descriptors_1 = sift_descriptors_1 , descriptors_2 = sift_descriptors_2;
    //cout<<"use SIFT to futher calculate the Feature Matching"<<endl;
    //--------------------------------------------------------------------------
    
    //use SURF for saving the processing time
    keypoints_1 = surf_keypoints_1, keypoints_2 = surf_keypoints_2;
    descriptors_1 = surf_descriptors_1 , descriptors_2 = surf_descriptors_2;
    cout<<"use SURF to futher calculate the Feature Matching"<<endl;
    
    
    // cout<<endl<<" ---Task 1.2. Feature Matching---"<<endl;
    std::vector< DMatch > myMatching;
    std::vector< DMatch > bestMatches;
    
    //self implement matching algorithm (Using left-right check)
    myMatching = matchingFunc( descriptors_1,descriptors_2);
    cout<<"number of point matched: "<<myMatching.size()<<endl;
    cout<<" surf matches"<<endl;
    
    Mat img_matches1;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, myMatching, img_matches1, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    imshow( "my Matching", img_matches1 );
    
    int size_goodpoint = 30;
    if( myMatching.size() < 30 )
    {
        size_goodpoint = (int)myMatching.size();
    }
    //sort the diestance in Dmatch
    std::sort(myMatching.begin(), myMatching.end());
    bestMatches = myMatching;
    bestMatches.resize(size_goodpoint);
    
    std::vector<KeyPoint> goodkeypoints_1, goodkeypoints_2;
    for(int d = 0 ; d< bestMatches.size();d++)
    {
        goodkeypoints_1.push_back(cv::KeyPoint( keypoints_1[bestMatches[d].queryIdx].pt, 1.f));
        goodkeypoints_2.push_back(cv::KeyPoint(keypoints_2[bestMatches[d].trainIdx].pt,1.f));
    }
    
    int ptCount = (int)bestMatches.size();
    Mat p1(ptCount, 2, CV_32F);
    Mat p2(ptCount, 2, CV_32F);
    std::vector< Point2f > po1;
    std::vector< Point2f > po2;
    
    
    Point2f pt;
    for (int i=0; i<ptCount; i++)
    {
        pt = keypoints_1[bestMatches[i].queryIdx].pt;
        p1.at<float>(i, 0) = pt.x;
        po1.push_back(Point2f(pt.x,pt.y));
        p1.at<float>(i, 1) = pt.y;
        
        pt = keypoints_2[bestMatches[i].trainIdx].pt;
        p2.at<float>(i, 0) = pt.x;
        po2.push_back(Point2f(pt.x,pt.y));
        p2.at<float>(i, 1) = pt.y;
    }
    
    //Task 1.3. Epipolar Geometry
    
    Mat m_Fundamental =findFundamental(p1, p2,size_goodpoint);
    cout<<"the fundamental matrix calculated:  "<<endl<<m_Fundamental<<endl;
    
    Mat m_Fundamental2 =findFundamental(p1.rowRange(0,7), p2.rowRange(0,7),7);
    
    string f2 = "EpipolarLines";
    cout<<"using my own fundamental matrix to draw Epipolar Lines"<<endl<<endl;
    cout<<"Press any button on imshow window to proceed"<<endl<<endl;
    
    //Task 1.4. Sparse 3D Points
    string file = "output.ply";
    outputPly(file, img_1, keypoints_1, img_2, keypoints_2, myMatching );
    
    cout<<"The file "<<file<<" was generated"<<endl;
    
    waitKey(0);
}

int main( int argc, char** argv )
{
    cout<<"COMP5421 Project 1 - 3D Reconstruction"<<endl;
    
    Mat img_1, img_2;

    cout<<"Please input the first image file name (without .png):"<<endl;
    string s1,s2;
    getline(cin, s1);
    img_1 = imread( s1+".png", IMREAD_GRAYSCALE );
    
    if( !img_1.data )
    {
        cout<<"cannot read the image"<<endl;
        return -1;
    }
    
    cout<<"Please input the second image file name (without .png):"<<endl;
    getline(cin, s2);
    
    img_2 = imread( s2+".png", IMREAD_GRAYSCALE );
    
    if( !img_2.data )
    {
        cout<<"cannot read the image"<<endl;
        return -1;
    }

        
    findFundamentalMatrices(img_1, img_2);

}

void outputPly( String file, Mat img_1, vector<KeyPoint> m_LeftKey, Mat img_2, vector<KeyPoint> m_RightKey, vector<DMatch> m_Matches)
{
    std::vector< DMatch > good_matches;
    int ptCount = (int)m_Matches.size();
    Mat p1(ptCount, 2, CV_32F);
    Mat p2(ptCount, 2, CV_32F);
    
    Point2f pt;
    for (int i=0; i<ptCount; i++)
    {
        pt = m_LeftKey[m_Matches[i].queryIdx].pt;
        p1.at<float>(i, 0) = pt.x;
        p1.at<float>(i, 1) = pt.y;
        
        pt = m_RightKey[m_Matches[i].trainIdx].pt;
        p2.at<float>(i, 0) = pt.x;
        p2.at<float>(i, 1) = pt.y;
    }
    
    Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;
    m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
    
    int OutlinerCount = 0;
    for (int i=0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] == 1)
        {
            good_matches.push_back(m_Matches[i]);
        }
        else
        {
            OutlinerCount++;
        }
    }
    
    vector<Point2f> m_LeftInlier;
    vector<Point2f> m_RightInlier;
    vector<DMatch> m_InlierMatches;
    
    int InlinerCount = ptCount - OutlinerCount;
    m_InlierMatches.resize(InlinerCount);
    m_LeftInlier.resize(InlinerCount);
    m_RightInlier.resize(InlinerCount);
    InlinerCount = 0;
    for (int i=0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
            m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
            m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
            m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
            m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
            m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
            InlinerCount++;
        }
    }
    
    Mat R,T;
    rotation(m_Fundamental,R,T,0);
    
    cv::hconcat(R, T, R);
    
    Mat P1 = Mat::eye(3, 4, CV_64F);
    
    Mat pnts4D;
    
    Mat P = cameraMatrix() * P1;
    
    cout<<endl<<"The projection matrix for camera 1"<<endl<<P<<endl;
    Mat P2 = cameraMatrix()*R;
    cout<<endl<<"The projection matrix for camera 2"<<endl<<P2<<endl;
    
    triangulatePoints(P, P2,m_LeftInlier, m_RightInlier, pnts4D );
    
    Mat dd = pnts4D.t();
    for(int i = 0; i < dd.rows; i++)
    {
        if(dd.at<float>(i,3) < 0.0)
        {
            cout<<"find new"<<endl;
            rotation(m_Fundamental,R,T,1);
            
            cv::hconcat(R, T, R);
            Mat P2 = cameraMatrix()*R;
            triangulatePoints(P, P2,m_LeftInlier, m_RightInlier, pnts4D );
            break;
        }
        
    }
    
    Mat pnts3D;
    convertPointsFromHomogeneous(pnts4D.t() , pnts3D);
    
    viz::writeCloud(file,pnts3D);
}

void rotation(Mat F, Mat &R, Mat &T, int r1)
{
    Mat A = cameraMatrix();
    
    Mat E = A.t() * F * A;
    
    SVD decomp = SVD(E,4);
    
    Mat U = decomp.u;
    
    Mat V = decomp.vt;
    
    Mat W(3, 3, CV_64F, Scalar(0));
    W.at<double>(0, 1) = -1;
    W.at<double>(1, 0) = 1;
    W.at<double>(2, 2) = 1;
    
    Mat R1 =  U * W * V;
    Mat R2 =  U * W.t() * V;
    
    
    Mat T1 = U.col(2);
    Mat T2 = -1* T1;
    
    if(r1 == 0)
    {
        R = R1;
    }
    else{
        R = R2;
    }
    
    cout<<endl<<"The rotational matrix"<<endl<<R<<endl;
    T = T2;
    cout<<endl<<"The translation vector"<<endl<<T<<endl;
    Mat test = Mat(4,1, CV_64F, Scalar(100));

}

vector< DMatch > matchingFunc(  Mat descriptors_1, Mat descriptors_2)
{
    std::vector< DMatch > good_matches;
    
    double** min_distance = (double **) malloc(descriptors_1.rows * sizeof(double *));
    for(int w = 0; w < descriptors_1.rows; w++)
    {
        min_distance[w] = (double *) malloc(2* sizeof(double));
        
        for(int i = 0; i < descriptors_2.rows; i++)
        {
            double norm_dist =  cv::norm(descriptors_1.row(w)-descriptors_2.row(i));
            if(i == 0)
            {
                min_distance[w][0] = norm_dist;
                min_distance[w][1] = i;
            }
            else if( min_distance[w][0] > norm_dist)
            {
                min_distance[w][0] = norm_dist;
                min_distance[w][1] = i;
            }
        }
        //check is that the optimal solution
        double check_min[2];
        for(int i = 0; i < descriptors_1.rows; i++)
        {
            double norm_dist =  cv::norm(descriptors_1.row(i)-descriptors_2.row(min_distance[w][1]));
            
            if(i == 0)
            {
                check_min[0] = norm_dist;
                check_min[1] = i;
            }
            else if( check_min[0] > norm_dist)
            {
                check_min[0] = norm_dist;
                check_min[1] = i;
            }
        }
        if(check_min[1] != w )
        {
            min_distance[w][0] = 0;
            min_distance[w][1] = -1;
        }
        else{
            if(min_distance[w][0] < 0.35)
            {
                good_matches.push_back( DMatch(w,  min_distance[w][1] , min_distance[w][0]));
            }
        }
    }
    free(min_distance);
    return good_matches;
}

Mat findFundamental(Mat k1, Mat k2, int N)
{
    Mat F =  Mat(3,3, CV_64F, double(0));
    //8 point algorithm
    if(N > 7)
    {
        cout<<"8 point algorithm"<<endl<<endl;
        Mat k1x = k1.col(0);
        Mat k1y = k1.col(1);
        Mat k2x = k2.col(0);
        Mat k2y = k2.col(1);
        
        Mat A  = Mat(N,9, CV_64F, double(1));
        
        Mat A1 = k1x.mul(k2x).t();
        Mat A2 = k1x.mul(k2y).t();
        Mat A3 = k1x.mul(1).t();
        Mat A4 = k1y.mul(k2x).t();
        Mat A5 = k1y.mul(k2y).t();
        Mat A6 = k1y.mul(1).t();
        Mat A7 = k2x.mul(1).t();
        Mat A8 = k2y.mul(1).t();
        
        Mat B;
        B.push_back(A1);
        B.push_back(A2);
        B.push_back(A3);
        B.push_back(A4);
        B.push_back(A5);
        B.push_back(A6);
        B.push_back(A7);
        B.push_back(A8);
        B.push_back(A1);
        
        A = B;
        
        A.row(8).setTo(1);
        
        Mat S,U,Vt;
        SVD::compute(A.t(),S,U,Vt);
        
        for(int i = 0; i< 3; i++)
        {
            F.at<double>(i,0)= Vt.at<float>(8,i*3);
            F.at<double>(i,1)= Vt.at<float>(8,i*3+1);
            F.at<double>(i,2)= Vt.at<float>(8,i*3+2);
            
        }
        SVD::compute(F,S,U,Vt);
        
        Mat s1 = Mat::eye(3,3, CV_64F);
        
        s1.at<double>(0,0) = S.at<double>(0,0);
        s1.at<double>(1,1) = S.at<double>(1,0);
        s1.at<double>(2,2) = 0.0;
        F = U * s1 * Vt;
        F = F.t();
        
    }
    else if(N == 7)//7 point algorithm
    {
        cout<<endl<<"7 point algorithm"<<endl;
        Mat k1x = k1.col(0);
        Mat k1y = k1.col(1);
        Mat k2x = k2.col(0);
        Mat k2y = k2.col(1);
        
        Mat A  = Mat(N,9, CV_64F, double(1));
        
        Mat A1 = k1x.mul(k2x).t();
        Mat A2 = k1x.mul(k2y).t();
        Mat A3 = k1x.mul(1).t();
        Mat A4 = k1y.mul(k2x).t();
        Mat A5 = k1y.mul(k2y).t();
        Mat A6 = k1y.mul(1).t();
        Mat A7 = k2x.mul(1).t();
        Mat A8 = k2y.mul(1).t();
        
        Mat B;
        B.push_back(A1);
        B.push_back(A2);
        B.push_back(A3);
        B.push_back(A4);
        B.push_back(A5);
        B.push_back(A6);
        B.push_back(A7);
        B.push_back(A8);
        B.push_back(A1);
        
        A = B;
        
        A.row(8).setTo(1);
        
        Mat S,U,Vt;
        SVD::compute(A.t(),S,U,Vt,4);
        
        Mat FF1 = Mat(3,3, CV_64F, double(0));
        Mat FF2 =  Mat(3,3, CV_64F, double(0));
     
        for(int i = 0; i< 3; i++)
        {
            FF1.at<double>(i,0)= Vt.at<float>(8,i*3);
            FF1.at<double>(i,1)= Vt.at<float>(8,i*3+1);
            FF1.at<double>(i,2)= Vt.at<float>(8,i*3+2);
            
            FF2.at<double>(i,0)= Vt.at<float>(7,i*3);
            FF2.at<double>(i,1)= Vt.at<float>(7,i*3+1);
            FF2.at<double>(i,2)= Vt.at<float>(7,i*3+2);
            
        }
        
        Mat FFF[2];
        FFF[0] = FF1;
        FFF[1] = FF2;
        
        double D[2][2][2];
        
        for(int x = 0; x<2; x++)
        {
            for(int y = 0; y<2; y++)
            {
                for(int  z= 0; z<2; z++)
                {
                    Mat Z = FFF[x].col(0);
                    
                    cv::hconcat(Z, FFF[y].col(1),Z);
                    cv::hconcat(Z, FFF[z].col(2),Z);
                    
                    D[x][y][z] = determinant(Z);
                    
                }
            }
        }
        
        double a[4] =  { D[1][1][1],
            D[1][1][0]+D[0][1][1]+D[1][0][1]-3*D[1][1][1],
            D[0][0][1]-2*D[0][1][1]-2*D[1][0][1]+D[1][0][0]-2*D[1][1][0]+D[0][1][0]+3*D[1][1][1],
            -D[1][0][0]+D[0][1][1]+D[0][0][0]+D[1][1][0]+D[1][0][1]-D[0][1][0]-D[0][0][1]-D[1][1][1]};
        
        Mat input = Mat(4,1,CV_64F,&a);
        
        cout<<"coefficients:"<<endl<<input<<endl;
        Mat result;
        solvePoly(input,result);
        
        cout<<"roots"<<endl<<result<<endl;
        
        Mat F1 = result.at<double>(2,0) * FF1 + (1 - result.at<double>(2,0)) * FF2;
        Mat F2 = result.at<double>(2,0) * FF1 + (1 - result.at<double>(2,0)) * FF2;
        Mat F3 = result.at<double>(2,0) * FF1 + (1 - result.at<double>(2,0)) * FF2;
        
        cout<<endl<<"result of 7-point algorthm"<<endl<<F1<<endl<<F2<<endl<<F3<<endl;
        F = F1;
        
    }
    return F;
    
}

Mat cameraMatrix()
{
    return(Mat_<double>(3,3) << 2759.48, 0,1520.69,0, 2764.16, 1006.81, 0, 0,1);
}
