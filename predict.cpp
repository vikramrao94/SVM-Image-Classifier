#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <dirent.h>
#include <vector>
#include <iostream>
#include <map>
#include <chrono>
#include "nlohmann/json.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using json = nlohmann::json;

map<string,string> hashTable;

/*
----------- Utilities -----------
*/

auto startTime(){
  return(chrono::steady_clock::now());
}

float timeElapsed(auto start){
  auto end = chrono::steady_clock::now();
  auto diff = end - start;
  return(chrono::duration <double, milli> (diff).count());
}

void listFiles(const string &path, vector<string> &cb) {
    if (auto dir = opendir(path.c_str())) {
        while (auto f = readdir(dir)) {
            if (!f->d_name || f->d_name[0] == '.') continue;
            if (f->d_type == DT_DIR)
                listFiles(path + f->d_name + "/", cb);

            if (f->d_type == DT_REG)
                cb.push_back(path + f->d_name);
        }
        closedir(dir);
    }
}
vector<string> split(string str,string sep){
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    vector<string> arr;
    current=strtok(cstr,sep.c_str());
    while(current!=NULL){
        arr.push_back(current);
        current=strtok(NULL,sep.c_str());
    }
    return arr;
}

//-----------------------------------------------------

vector<KeyPoint> detectKeypoint(Mat img){
  int minHessian = 400;
  Ptr<SURF> detector = SURF::create(minHessian);
  //Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(/*minHessian*/);
  vector<KeyPoint> keypoints;
  detector->detect( img, keypoints );
  return keypoints;
}

Mat describeKeypoint(Mat img, vector<KeyPoint> kp){
  Mat res;
  Ptr<SURF> descriptor = SURF::create();
  descriptor->compute ( img, kp, res );
  return res;
}

Mat loadVocab(){
  FileStorage fileStorage("dictionary.yml",cv::FileStorage::READ);
  Mat vocabulary;
  fileStorage["vocabulary"]>>vocabulary;
  fileStorage.release();
  cout<<"dictionary loaded"<<endl;
  return vocabulary;
}


Mat loadVocabulary(Mat predictImage,Mat vocab){


    Mat vocabulary=vocab,predictImageDescriptor;


    Ptr<DescriptorExtractor> extractor=SURF::create();
    Ptr<DescriptorMatcher> flannDescriptorMatcher(new FlannBasedMatcher);

    BOWImgDescriptorExtractor bowExtractor(extractor,flannDescriptorMatcher);
    bowExtractor.setVocabulary(vocabulary);
    vector<KeyPoint> predictKeypoint=detectKeypoint(predictImage);
    if(predictKeypoint.empty())
        cout<<"no keypoint "<<endl;
    bowExtractor.compute(predictImage,predictKeypoint,predictImageDescriptor);
    //cout<<"load vocabulary return"<<endl;
    if(predictImageDescriptor.empty())
        cout<<"no descriptor extracted"<<endl;
    return predictImageDescriptor;

}

Mat predict(string imageName,Mat vocab){
    cout<<"predict"<<endl;
    Mat image=imread(imageName), predictMat;
    //cout<<image.type();
    resize(image, image, Size(480,640), CV_INTER_LINEAR);
    cvtColor(image, predictMat, CV_BGR2GRAY);
    if(predictMat.empty())
        cout<<"image is not read"<<endl;
    Ptr<SVM> svm=Algorithm::load<SVM>("svmtrained.yml");
    cout<<"svm loaded successfully"<<endl;
    Mat predictImageDescriptor=loadVocabulary(predictMat,vocab);
    int result=(int)svm->predict(predictImageDescriptor);
    ifstream ifs("labels.json");
    json j = json::parse(ifs);

    string res=j.at(to_string(result));
    //cout<<"hash: "<<hashTable[res]<<endl;
    Mat matchedImage=imread(hashTable[res]);
    Mat combined(1280,960,CV_8UC3);
    matchedImage.copyTo(combined(Rect(0,200,matchedImage.size().width,matchedImage.size().height)));
    image.copyTo(combined(Rect(matchedImage.size().width+100,200,image.size().width,image.size().height)));
    //hconcat(image,matchedImage,matchedImage);

    cout<<"Result: "<<res<<endl;

    return combined;

}



int main(){
  vector<string> templates;
  string dir="FULL";
  listFiles(dir+"/",templates);
  for(int i=0;i<templates.size();i++){
    hashTable[split(templates[i],"/")[2]]=templates[i];
  }
  Mat vocab=loadVocab();

  string testDir="Sleeves";
  vector<string> testImages;
  listFiles(testDir+"/",testImages);
  for(int i=0;i<testImages.size();i++){
    auto start=startTime();
    Mat result=predict(testImages[i],vocab);
    string timeTaken="Time: "+to_string(timeElapsed(start))+" ms";
    putText(result,
            timeTaken,
            Point(20,50), // Coordinates
            FONT_HERSHEY_COMPLEX_SMALL, // Font
            1.0, // Scale. 2.0 = 2x bigger
            Scalar(255,255,255), // BGR Color
            1, // Line Thickness (Optional)
            CV_AA);
    imwrite("out/"+to_string(i)+".png",result);
  }
  return 0;
}
