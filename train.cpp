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

int length;
int typeImage;
int labelSize;
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
  detector->detect(img,keypoints);
  return keypoints;
}

Mat describeKeypoint(Mat img, vector<KeyPoint> kp){
  Mat res;
  Ptr<SURF> descriptor = SURF::create();
  descriptor->compute (img,kp,res);
  return res;
}

Mat getDescriptors(string dir){
  vector<string> templates;
  listFiles(dir+"/",templates);
  Mat training_descriptors;
  int i;
  for(i=0;i<templates.size();i++){
    //
    Mat ref=imread(templates[i],IMREAD_GRAYSCALE);
    vector<KeyPoint> kp_ref=detectKeypoint(ref);
    Mat des_ref=describeKeypoint(ref,kp_ref);
    training_descriptors.push_back(des_ref);

    cout<<split(templates[i],"/")[2]<<" complete"<<endl;
  }
//  cout<<"count: "<<i-1<<endl;
  //cout<<"diagnostics: "<<training_descriptors.size()<<endl;
  return training_descriptors;
}

void trainBOW(string dir){
  cout<<"BoW training"<<endl;
  int dictionarySize = 1000;
  TermCriteria tc(cv::TermCriteria::COUNT,100,0.001);
  //retries number
  int retries=1;
  //necessary flags
  int flags=KMEANS_PP_CENTERS;

  Mat unclustered_descriptors=getDescriptors(dir);
  cout<<"diagnostics: "<<unclustered_descriptors.size();
  Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
  BOWKMeansTrainer bowtrainer(dictionarySize,tc,retries,flags); //num clusters
  bowtrainer.add(unclustered_descriptors);
  Mat vocabulary = bowtrainer.cluster();

  FileStorage fs("dictionary.yml", FileStorage::WRITE);
  fs << "vocabulary" << vocabulary;
  fs.release();
  cout<<"BoW clustering complete"<<endl;
}

map<string,Mat> createTrainingImage(vector<string> filenames,vector<string> labels,Mat vocabulary){

  cout<<"Classifier phase entered"<<endl;

  Mat trainingImg;
  Mat image;
  vector<KeyPoint> kp;
  Mat descriptor;
  string label;
  map<string,Mat> mapedImage;

  Ptr<DescriptorExtractor> extractor=SURF::create();
  Ptr<DescriptorMatcher> flannDescriptorMatcher(new FlannBasedMatcher);

  BOWImgDescriptorExtractor bowExtractor(extractor,flannDescriptorMatcher);
  bowExtractor.setVocabulary(vocabulary);

  cout<<"Initialized BOW vocabularies..."<<endl;

  cout<<"Total number of files for trainig: "<<filenames.size()<<endl;
  cout<<"Total number of labels: "<<labels.size()<<endl;

  labelSize=labels.size();


  for(int i=0;i<filenames.size();i++){

    image=imread(filenames[i],IMREAD_GRAYSCALE);
    kp=detectKeypoint(image);
    bowExtractor.compute(image,kp,descriptor);
    //cout<<"descriptor size: "<<descriptor.size();
    mapedImage[labels[i]].create(0,descriptor.cols,descriptor.type());
    mapedImage[labels[i]].push_back(descriptor);
  }

  length=descriptor.cols;
  typeImage=descriptor.type();
  //cout<<"type: "<<typeImage<<endl;
  cout<<"Extracted BoW descriptors"<<endl;
  return mapedImage;
}

void trainSVMFinal(map<string,Mat> mapedImage){
  cout<<"Train SVM final stages"<<endl;

  Mat svmTrainImage;
  Ptr<SVM> svm = SVM::create();
  Mat labelsMat(labelSize,1, CV_32S);
  Mat samples(0,length,typeImage);
  string stringLabel;
  int i=0;
  json out;
  for(map<string,Mat>::iterator it=mapedImage.begin();it!=mapedImage.end();it++){
    stringLabel=it->first;
    labelsMat.at<int>(i)=i;
    samples.push_back(mapedImage[stringLabel]);
    out[to_string(i)]=it->first;
    i++;
  }
  cout<<"diagnostics: "<<samples.size()<<endl;
  //cout<<labelsMat<<endl;
  ofstream file("labels.json");
  file << setw(4) << out << endl;
  cout<<"Successfully labeled images"<<endl;

  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::LINEAR);

  //TermCriteria term(cv::TermCriteria::COUNT,100,1e-6);
  //svmParam.term_crit=term;
  cout<<"Parameters created"<<endl;

  svm->train(samples,ROW_SAMPLE,labelsMat );
  cout<<"trained successfully "<<endl;
  string fs="svmtrained.yml";
  svm->save(fs);

}

void trainSVM(string trainingDir){
  FileStorage fileStorage("dictionary.yml",cv::FileStorage::READ);
  Mat vocabulary;
  fileStorage["vocabulary"]>>vocabulary;
  fileStorage.release();

  vector<string> filenames;
  vector<string> labels;

  listFiles(trainingDir+"/",filenames);
  for (int i=0;i<filenames.size();i++){
    labels.push_back(split(filenames[i],"/")[2]);
  }
  cout<<"Labels added successfully"<<endl;

  map<string,Mat> mapedImage=createTrainingImage(filenames,labels,vocabulary);
  trainSVMFinal(mapedImage);


}
int main()
{
  string trainingDir="training";
  trainBOW(trainingDir);
  trainSVM(trainingDir);
  return 0;
}
