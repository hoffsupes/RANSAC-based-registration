#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <cstdarg>
#include "opencv2/opencv.hpp"
#include "fstream"
#include <dirent.h>
#include <math.h>
#include <time.h>
#include <opencv2/videostab.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <memory>

#define NORM_CPP(x) sqrt(pow(x,2))

using namespace std;
using namespace cv;

Mat getreducemat(Mat mgpts,Mat scpts,Mat a13,Mat a46)
{

	Mat red,A13,A46,newmg,newsc,XX,YY;
	
	Mat SCP[] = {scpts,Mat::ones(scpts.rows,1,scpts.type()) };
	hconcat(SCP,2,newsc);
	
	repeat(a13, 1,scpts.rows, A13);	A13 = A13.t();
	repeat(a46,1,scpts.rows, A46);	A46 = A46.t();
	
	multiply(A13,newsc,A13);
	multiply(A46,newsc,A46);
	
	reduce(A13,XX,1,CV_REDUCE_SUM);
	reduce(A46,YY,1,CV_REDUCE_SUM);

	Mat rree[] = {XX,YY};
	hconcat(rree,2,red);
	
	red = mgpts - red;
	
	return red;
	
}

Mat get_affine_matrix(Mat movpoints,Mat srcpoints)			// srcpoints = moving frame ||| movpoints = static frame
// Mat get_affine_matrix(Mat srcpoints,Mat movpoints)			// srcpoints = moving frame ||| movpoints = static frame
{
float epsilon = 0.01;
Mat ontm = Mat::ones(srcpoints.rows, 1, srcpoints.type());
Mat A,X,Y,a13, a46;
	
Mat mtarr[] = {srcpoints, ontm};
hconcat(mtarr, 2 , A);

movpoints.col(0).copyTo(X);
movpoints.col(1).copyTo(Y);

/// FOR TESTING PURPOSES ONLY REMOVE LATER	

 a13 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * X);		// Check for SVD parameters for accurate warp
 a46 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * Y);

// a13 = (A.inv(DECOMP_SVD)) * X;
// a46 = (A.inv(DECOMP_SVD)) * Y;
float J = 0,Jold = 0,delJ = 0,olddelJ = 0;
float thep = 100;
int cco = 0;	
	Mat Xresidue, Yresidue,XW,YW;

	// loop condition to be put here
	do{

	Mat redu =  getreducemat(movpoints,srcpoints,a13,a46);
	Mat W;	
	divide(1,abs(redu) + epsilon,W);
	
	multiply(A.col(0),W.col(0),A.col(0));
	multiply(A.col(1),W.col(1),A.col(1));
	
	multiply(X,W.col(0),X);
	multiply(Y,W.col(1),Y);
	
 	a13 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * X);		// Check for SVD parameters for accurate warp
 	a46 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * Y);	
	
	pow(redu,2,redu);
		
	Jold = J;		
	J = sum(redu.col(0))[0] + sum(redu.col(1))[0];
	olddelJ = delJ;
	delJ = abs(J - Jold);
		
	cco++;	
	}while(	(delJ > thep) && (cco > 0) & (( (abs(delJ - olddelJ) == 0)) || (abs(delJ - olddelJ) <= 1.5)) ) ;

Mat affine_matrix; Mat tmpone  = Mat::zeros(1,3,CV_32F); tmpone.at<float>(2) = 1;
affine_matrix.push_back(a13.t());
affine_matrix.push_back(a46.t());
//affine_matrix.push_back(tmpone);

	
return affine_matrix;
}

Mat getOptimumTform_distance_criteria(Mat movpoints, Mat srcpoints)					// Using distance criteria
{
    int nsamples = 60;
    int siz;
    int N = 100;
    float r = 0;
    int best_n = 0;
    float dist_th = 9;
	float r_thresh = 0.8;
	
    Mat tform = Mat::eye(2,3,CV_32F);
    Mat nmov = movpoints.t(); nmov.push_back(Mat::ones(1,nmov.cols,nmov.type()));
    Mat nsrc = srcpoints.t(); nsrc.push_back(Mat::ones(1,nsrc.cols,nsrc.type()));
    
    
//     siz = ( (movpoints.rows() > movpoints.cols)? movpoints.rows:movpoints.cols );
    siz = movpoints.rows;
    int j =0;
   // for(int j = 0; j< N; j++)
    while( (r < r_thresh) || ( j <= N) )
    {
        Mat tmovp;
        Mat tsrcp;
        
        for(int i = 0; i< nsamples; i++)
        {
        int ransam = rand() % (siz-1);			
//         sam.push_back(ransam);
        tmovp.push_back(movpoints.row(ransam));
        tsrcp.push_back(srcpoints.row(ransam));
        }

        Mat get_affin = get_affine_matrix(tmovp,tsrcp);
        Mat wmov = get_affin * nmov ;
        Mat WNmov; 
        WNmov.push_back(wmov.row(0));
        WNmov.push_back(wmov.row(1)); wmov.release();
        WNmov = WNmov.t();
        WNmov -= srcpoints;
		

		
        multiply(WNmov,WNmov,WNmov);
        Mat rs;
        reduce(WNmov,rs,1,CV_REDUCE_SUM);
		cv::sqrt(rs,rs);
		
		Mat U = (rs < dist_th);					// Is an inlier ???
			
        int n_inliers = countNonZero(U);		// Counting all true instances of the above condition
        
        if(n_inliers > best_n)    
        {
            best_n = n_inliers;
            get_affin.copyTo(tform);
        }
        
        r = ( float(best_n) / float(siz) );
    j++;
		
		waitKey();
		waitKey();
		waitKey();
		waitKey();
		waitKey();
		waitKey();
    }
    
    Mat mv = tform*nmov;
    Mat MV; 
    MV.push_back(mv.row(0));
    MV.push_back(mv.row(1)); mv.release();
    MV = MV.t();
    
    MV -= srcpoints;
    multiply(MV,MV,MV);
    Mat rr;
    reduce(MV,rr,1,CV_REDUCE_SUM);
    cv::sqrt(rr,rr);
	
	
    Mat is_inlier = (rr < dist_th);
    
    vector<Point2f> Fmov,Fsrc;
    
    for(int i = 0; i < is_inlier.rows; i++)
    {
        if(is_inlier.at<uchar>(i))
        {
            Point2f mvp,srcp;
            
            mvp.x = movpoints.at<float>(i,0);
            mvp.y = movpoints.at<float>(i,1);
            
            srcp.x = srcpoints.at<float>(i,0);
            srcp.y = srcpoints.at<float>(i,1);
            
        Fmov.push_back(mvp);
        Fsrc.push_back(srcp);
        }
    }
    
    Mat final_tform = findHomography(Fmov,Fsrc,0);
    return final_tform;
}

Mat getOptimumTform(Mat movpoints, Mat srcpoints,Mat refgray, Mat movgray)			// Not using distance criteria but rather opticalFlow
{
    int nsamples = 60;
    int siz;
    int N = 100;
    float r = 0;
    int best_n = 0;
	float r_thresh = 0.8;
	
    Mat tform = Mat::eye(2,3,CV_32F);
    Mat nmov = movpoints.t(); nmov.push_back(Mat::ones(1,nmov.cols,nmov.type()));
    Mat nsrc = srcpoints.t(); nsrc.push_back(Mat::ones(1,nsrc.cols,nsrc.type()));
    
	vector<int> rand_vect;
	
	for(int i = 0; i < movpoints.rows; i++)	{rand_vect.push_back(i);}
	random_shuffle(rand_vect.begin(), rand_vect.end());
	
//     siz = ( (movpoints.rows() > movpoints.cols)? movpoints.rows:movpoints.cols );
    siz = movpoints.rows;
    int j = 0;
   // for(int j = 0; j< N; j++)
    while( (r < r_thresh) || ( j <= N) )
    {
//		cout<<"\n J::"<< j <<"\n";
        Mat tmovp;
        Mat tsrcp;
        int ransam = rand() % (movpoints.rows - nsamples - 1);			// random rand_vector position chosen 		
        
        for(int i = ransam; i< nsamples + ransam; i++)
        {
			
//			cout << " \n Random value:::: "<< rand_vect[i] <<"\n";
//         sam.push_back(ransam);
	    tmovp.push_back(movpoints.row( rand_vect[i] ));
        tsrcp.push_back(srcpoints.row( rand_vect[i] ));
        }

        Mat get_affin = get_affine_matrix(tmovp,tsrcp);
        Mat wmov = get_affin * nmov ;
        Mat WNmov; 
        WNmov.push_back(wmov.row(0));
        WNmov.push_back(wmov.row(1)); wmov.release();
        WNmov = WNmov.t();
		
		vector<Point2f> dstPf;
		vector<uchar> status;
		vector<float> err;
		vector<Point2f> srcPf;
		
		for(int i = 0; i < WNmov.rows; i++)
		{
			
		Point2f P;
		P.x = WNmov.at<float>(i,0);
		P.y = WNmov.at<float>(i,1);
		dstPf.push_back(P);	
			
			
		Point2f P1;
		P1.x = srcpoints.at<float>(i,0);
		P1.y = srcpoints.at<float>(i,1);
		srcPf.push_back(P1);	
		}
		

		calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err,Size(21,21),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.01),OPTFLOW_USE_INITIAL_FLOW);
		
		int n_inliers;
		
		for(int i = 0; i < status.size(); i++ ) {if(status[i]) { n_inliers++; } }
		
		/*
		WNmov -= srcpoints;
		multiply(WNmov,WNmov,WNmov);
        Mat rs;
        reduce(WNmov,rs,1,CV_REDUCE_SUM);

        int n_inliers = rs.rows - countNonZero(rs);
        
		*/
		
        if(n_inliers > best_n)    
        {
            best_n = n_inliers;
            get_affin.copyTo(tform);
        }
        
        r =  ( float(best_n) / float(siz) ) ;
		
    j++;
		
    }
    
    Mat mv = tform*nmov;
    Mat MV; 
    MV.push_back(mv.row(0));
    MV.push_back(mv.row(1)); mv.release();
    MV = MV.t();
    
			
		vector<Point2f> dstPf;
		vector<uchar> status;
		vector<float> err;
		vector<Point2f> srcPf;
		
		for(int i = 0; i < MV.rows; i++)
		{
			
		Point2f P;
		P.x = MV.at<float>(i,0);
		P.y = MV.at<float>(i,1);
		dstPf.push_back(P);	
			
			
		Point2f P1;
		P1.x = srcpoints.at<float>(i,0);
		P1.y = srcpoints.at<float>(i,1);
		srcPf.push_back(P1);	
		}
		

		calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err,Size(21,21),3,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.01),OPTFLOW_USE_INITIAL_FLOW);
		
		Mat is_inlier;
		
		for(int i = 0; i < status.size(); i++ ) {if(status[i]) { is_inlier.push_back(255); } else { is_inlier.push_back(0); } }
	
	/*
    MV -= srcpoints;
    multiply(MV,MV,MV);
    Mat rr;
    reduce(MV,rr,1,CV_REDUCE_SUM);
    
    Mat is_inlier = (rr == 0);
    */
	
    vector<Point2f> Fmov,Fsrc;
    
    for(int i = 0; i < is_inlier.rows; i++)
    {
        if(is_inlier.at<uchar>(i))
        {
            Point2f mvp,srcp;
            
            mvp.x = movpoints.at<float>(i,0);
            mvp.y = movpoints.at<float>(i,1);
            
            srcp.x = srcpoints.at<float>(i,0);
            srcp.y = srcpoints.at<float>(i,1);
            
        Fmov.push_back(mvp);
        Fsrc.push_back(srcp);
        }
    }
    
    Mat final_tform = findHomography(Fmov,Fsrc,0);
    return final_tform;
}

Mat register_fram_dynamic(Mat refframe, Mat movingframe)			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	if(!srcPf.size()){return movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = getOptimumTform(mvgpts,srcpts,refgray,movgray);
	}
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpPerspective(movingframe,warped_frame,warp_m, movingframe.size());
	return warped_frame;

}

Mat register_RANSAC(Mat refframe, Mat movingframe)			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	if(!srcPf.size()){return movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = getOptimumTform(mvgpts,srcpts,refgray,movgray);
	}
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpPerspective(movingframe,warped_frame,warp_m, movingframe.size());
	return warped_frame;

}

Mat register_RANSAC_euc(Mat refframe, Mat movingframe)			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	if(!srcPf.size()){return movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = getOptimumTform_distance_criteria(mvgpts,srcpts);
	}
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpPerspective(movingframe,warped_frame,warp_m, movingframe.size());
	return warped_frame;

}

Mat register_f(Mat refframe, Mat movingframe)			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf,10000, 0.01, 30);
	if(!srcPf.size()){return movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = findHomography(dstPf,srcPf,CV_RANSAC);
	}
	
	Mat nmov = mvgpts.t();
	nmov.push_back( Mat::ones(1,nmov.cols,nmov.type()) );
	warp_m.convertTo(warp_m,nmov.type());
	Mat WN = warp_m * nmov ;
	Mat WW; 
	WW.push_back(WN.row(0));
	WW.push_back(WN.row(1));	WN.release();
	WW = WW.t();
	WW -= srcpts;
	cv::pow(WW,2,WW);
	Mat RR; reduce(WW,RR,1,CV_REDUCE_SUM);
	Mat UU = RR < 1;
		
	vector<Point2f> mv;
	vector<Point2f> sr;
	
	Mat K = Mat::zeros(movingframe.size(), CV_32SC1);
	
	for(int i = 0; i < UU.rows; i++)
	{
	
	if(UU.at<uchar>(i))	
	{
	mv.push_back(dstPf[i]);	
	sr.push_back(srcPf[i]);
	}
		else
		{
		K.at<int>( dstPf[i].y, dstPf[i].x ) = 255;
		}
		
	}
	Mat KK ; movingframe.copyTo(KK);
//	watershed(KK,K);
	K.convertTo(K,CV_8UC1);
	imshow("K",K);
	
	if(mv.size() == 0 || sr.size() == 0 )
	{
	warp_m = Mat::eye(3,3,CV_32F);
	}
	else{
	warp_m = findHomography(mv,sr,0);
	}
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpPerspective(movingframe,warped_frame,warp_m, movingframe.size());
	return warped_frame;

}

Mat register_BG(Mat refframe, Mat movingframe)			// all rows and columns in an image
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	for(int i = 0;  i < refframe.rows;i++){for(int j = 0; j < refframe.cols; j++){ Point2f G; G.x = j; G.y = i; srcPf.push_back(G);  } }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = findHomography(dstPf,srcPf,CV_RANSAC);
	}
	
	Mat nmov = mvgpts.t();
	nmov.push_back( Mat::ones(1,nmov.cols,nmov.type()) );
	warp_m.convertTo(warp_m,nmov.type());
	Mat WN = warp_m * nmov ;
	Mat WW; 
	WW.push_back(WN.row(0));
	WW.push_back(WN.row(1));	WN.release();
	WW = WW.t();
	WW -= srcpts;
	cv::pow(WW,2,WW);
	Mat RR; reduce(WW,RR,1,CV_REDUCE_SUM);
	Mat UU = RR < 1;
		
	vector<Point2f> mv;
	vector<Point2f> sr;
	
	Mat K = Mat::zeros(movingframe.size(), CV_8UC1);
	
	for(int i = 0; i < UU.rows; i++)
	{
	
	if(UU.at<uchar>(i))	
	{
	mv.push_back(dstPf[i]);	
	sr.push_back(srcPf[i]);
	}
		else
		{
		K.at<uchar>( dstPf[i].y, dstPf[i].x ) = 255;
		}
		
	}
	
	Mat KK; movingframe.copyTo(KK,K);
	
	imshow("K",KK);
	
	if(mv.size() == 0 || sr.size() == 0 )
	{
	warp_m = Mat::eye(3,3,CV_32F);
	}
	else{
	warp_m = findHomography(mv,sr,0);
	}
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpPerspective(movingframe,warped_frame,warp_m, movingframe.size());
	return warped_frame;

}

void obtain_outliers(Mat warp_m,Mat mvgpts, Mat srcpts, vector<Point2f> & mv, vector<Point2f> & sr,vector<Point2f> dstPf, vector<Point2f> srcPf)
{

	Mat nmov = mvgpts.t();
	nmov.push_back( Mat::ones(1,nmov.cols,nmov.type()) );
	warp_m.convertTo(warp_m,nmov.type());
	Mat WN = warp_m * nmov ;
	Mat WW; 
	WW.push_back(WN.row(0));
	WW.push_back(WN.row(1));	WN.release();
	WW = WW.t();
	WW -= srcpts;
	cv::pow(WW,2,WW);
	Mat RR; reduce(WW,RR,1,CV_REDUCE_SUM);
	Mat UU = RR < 1;
	
	for(int i = 0; i < UU.rows; i++)
	{
	
	if(UU.at<uchar>(i) == 0)	
		{
		mv.push_back(dstPf[i]);	
		sr.push_back(srcPf[i]);
		}
		
	}
	
	
}

Mat clean_mask(Mat refframe, Mat movingframe, Mat oldmask)			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf,10000, 0.01, 30);
	if(!srcPf.size()){return movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = findHomography(dstPf,srcPf,CV_RANSAC);
	}
	vector<Point2f> mv,mv1,mv2;
	vector<Point2f> sr,sr1,sr2;
	
	obtain_outliers(warp_m,mvgpts,srcpts,mv,sr,dstPf,srcPf);

	for(int i = 0; i< mv.size(); i++)
	{
	
		if(oldmask.at<uchar>(mv[i].y,mv[i].x) != 0)
		{
		mv1.push_back(mv[i]);
		sr1.push_back(sr[i]);
		}
		
	}
	
	warp_m = findHomography(mv1,sr1,CV_RANSAC);
	
	Mat MV1,SR1;
	
	for(int i = 0; i< mv1.size(); i++) {Mat L = Mat::zeros(1,2,CV_32F); L.at<float>(0) = mv1[i].x;  L.at<float>(1) = mv1[i].y; MV1.push_back(mv1); }
	for(int i = 0; i< sr1.size(); i++) {Mat L = Mat::zeros(1,2,CV_32F); L.at<float>(0) = sr1[i].x;  L.at<float>(1) = sr1[i].y; SR1.push_back(sr1); }
	
	obtain_outliers(warp_m,MV1,SR1,mv2,sr2,mv1,sr1);
	
	Mat K = Mat::zeros(oldmask.size(), oldmask.type() );
	
	for(int i = 0;  i< mv2.size(); i++)
	{
	
		K.at<uchar>(mv2[i].y,mv2[i].x) = 255;
		
	}
	
	return K;
}

int main()
{

VideoCapture cap;
cap.open("/home/ml/shaky2.mp4");

VideoWriter wir;
int codec = CV_FOURCC('M','J','P','G');
wir.open("RANSAC_OPTFL.avi",codec ,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));	

VideoWriter wir2;
wir2.open("RANSAC_euc_OPTFL.avi",codec ,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));	
	
	
//	cap.set(CV_CAP_PROP_POS_MSEC,50000);
	
Mat G; cap >> G;

while( cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT) )
{
Mat F; cap>>F;
Mat Y = register_RANSAC(G,F);
Mat UI = register_RANSAC_euc(G,F);	
wir << Y ;
wir2 << UI ;	
F.copyTo(G);
}

	cout << "\nDone 	Writing RANSAC video\n";
	
return 1;
}