
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


/****************************************************************************************\
*                       Auxiliary functions declarations                                 *
\****************************************************************************************/

/* Generates a set of classes centers in quantity <num_of_clusters> that are generated as
   uniform random vectors in parallelepiped, where <data> is concentrated. Vectors in
   <data> should have horizontal orientation. If <centers> != NULL, the function doesn't
   allocate any memory and stores generated centers in <centers>, returns <centers>.
   If <centers> == NULL, the function allocates memory and creates the matrice. Centers
   are supposed to be oriented horizontally. */
CvMat* icvGenerateRandomClusterCenters( int seed,
                                        const CvMat* data,
                                        int num_of_clusters,
                                        CvMat* centers CV_DEFAULT(0));

/* Fills the <labels> using <probs> by choosing the maximal probability. Outliers are
   fixed by <oulier_tresh> and have cluster label (-1). Function also controls that there
   weren't "empty" clusters by filling empty clusters with the maximal probability vector.
   If probs_sums != NULL, filles it with the sums of probabilities for each sample (it is
   useful for normalizing probabilities' matrice of FCM) */
void icvFindClusterLabels( const CvMat* probs, float outlier_thresh, float r,
                           const CvMat* labels );

typedef struct CvSparseVecElem32f
{
    int idx;
    float val;
}
CvSparseVecElem32f;

/* Prepare training data and related parameters */
#define CV_TRAIN_STATMODEL_DEFRAGMENT_TRAIN_DATA    1
#define CV_TRAIN_STATMODEL_SAMPLES_AS_ROWS          2
#define CV_TRAIN_STATMODEL_SAMPLES_AS_COLUMNS       4
#define CV_TRAIN_STATMODEL_CATEGORICAL_RESPONSE     8
#define CV_TRAIN_STATMODEL_ORDERED_RESPONSE         16
#define CV_TRAIN_STATMODEL_RESPONSES_ON_OUTPUT      32
#define CV_TRAIN_STATMODEL_ALWAYS_COPY_TRAIN_DATA   64
#define CV_TRAIN_STATMODEL_SPARSE_AS_SPARSE         128

int
cvPrepareTrainData( const char* /*funcname*/,
                    const CvMat* train_data, int tflag,
                    const CvMat* responses, int response_type,
                    const CvMat* var_idx,
                    const CvMat* sample_idx,
                    bool always_copy_data,
                    const float*** out_train_samples,
                    int* _sample_count,
                    int* _var_count,
                    int* _var_all,
                    CvMat** out_responses,
                    CvMat** out_response_map,
                    CvMat** out_var_idx,
                    CvMat** out_sample_idx=0 );

void
cvSortSamplesByClasses( const float** samples, const CvMat* classes,
                        int* class_ranges, const uchar** mask CV_DEFAULT(0) );

void
cvCombineResponseMaps (CvMat*  _responses,
                 const CvMat*  old_response_map,
                       CvMat*  new_response_map,
                       CvMat** out_response_map);

void
cvPreparePredictData( const CvArr* sample, int dims_all, const CvMat* comp_idx,
                      int class_count, const CvMat* prob, float** row_sample,
                      int as_sparse CV_DEFAULT(0) );

/* copies clustering [or batch "predict"] results
   (labels and/or centers and/or probs) back to the output arrays */
void
cvWritebackLabels( const CvMat* labels, CvMat* dst_labels,
                   const CvMat* centers, CvMat* dst_centers,
                   const CvMat* probs, CvMat* dst_probs,
                   const CvMat* sample_idx, int samples_all,
                   const CvMat* comp_idx, int dims_all );
#define cvWritebackResponses cvWritebackLabels

#define XML_FIELD_NAME "_name"
CvFileNode* icvFileNodeGetChild(CvFileNode* father, const char* name);
CvFileNode* icvFileNodeGetChildArrayElem(CvFileNode* father, const char* name,int index);
CvFileNode* icvFileNodeGetNext(CvFileNode* n, const char* name);


void cvCheckTrainData( const CvMat* train_data, int tflag,
                       const CvMat* missing_mask,
                       int* var_all, int* sample_all );

CvMat* cvPreprocessIndexArray( const CvMat* idx_arr, int data_arr_size, bool check_for_duplicates=false );

CvMat* cvPreprocessVarType( const CvMat* type_mask, const CvMat* var_idx,
                            int var_all, int* response_type );

CvMat* cvPreprocessOrderedResponses( const CvMat* responses,
                const CvMat* sample_idx, int sample_all );

CvMat* cvPreprocessCategoricalResponses( const CvMat* responses,
                const CvMat* sample_idx, int sample_all,
                CvMat** out_response_map, CvMat** class_counts=0 );

const float** cvGetTrainSamples( const CvMat* train_data, int tflag,
                   const CvMat* var_idx, const CvMat* sample_idx,
                   int* _var_count, int* _sample_count,
                   bool always_copy_data=false );

namespace cv
{
    struct DTreeBestSplitFinder
    {
        DTreeBestSplitFinder(){ tree = 0; node = 0; }
        DTreeBestSplitFinder( CvDTree* _tree, CvDTreeNode* _node);
        DTreeBestSplitFinder( const DTreeBestSplitFinder& finder, Split );
        virtual ~DTreeBestSplitFinder() {}
        virtual void operator()(const BlockedRange& range);
        void join( DTreeBestSplitFinder& rhs );
        Ptr<CvDTreeSplit> bestSplit;
        Ptr<CvDTreeSplit> split;
        int splitSize;
        CvDTree* tree;
        CvDTreeNode* node;
    };

    struct ForestTreeBestSplitFinder : DTreeBestSplitFinder
    {
        ForestTreeBestSplitFinder() : DTreeBestSplitFinder() {}
        ForestTreeBestSplitFinder( CvForestTree* _tree, CvDTreeNode* _node );
        ForestTreeBestSplitFinder( const ForestTreeBestSplitFinder& finder, Split );
        virtual void operator()(const BlockedRange& range);
    };
}


class CV_EXPORTS_W DorniSVM : public CvSVM
{
public:
    // SVM type
    enum { C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104 };

    // SVM kernel type
    enum { LINEAR=0, POLY=1, RBF=2, SIGMOID=3 };

    // SVM params type
    enum { C=0, GAMMA=1, P=2, NU=3, COEF=4, DEGREE=5 };

    CV_WRAP DorniSVM();
    virtual ~DorniSVM();

    DorniSVM( const CvMat* trainData, const CvMat* responses,
           const CvMat* varIdx=0, const CvMat* sampleIdx=0,
           CvSVMParams params=CvSVMParams() );

    virtual bool train( const CvMat* trainData, const CvMat* responses,
                        const CvMat* varIdx=0, const CvMat* sampleIdx=0,
                        CvSVMParams params=CvSVMParams() );

    virtual bool train_auto( const CvMat* trainData, const CvMat* responses,
        const CvMat* varIdx, const CvMat* sampleIdx, CvSVMParams params,
        int kfold = 10,
        CvParamGrid Cgrid      = get_default_grid(DorniSVM::C),
        CvParamGrid gammaGrid  = get_default_grid(DorniSVM::GAMMA),
        CvParamGrid pGrid      = get_default_grid(DorniSVM::P),
        CvParamGrid nuGrid     = get_default_grid(DorniSVM::NU),
        CvParamGrid coeffGrid  = get_default_grid(DorniSVM::COEF),
        CvParamGrid degreeGrid = get_default_grid(DorniSVM::DEGREE),
        bool balanced=false );

    virtual float predict( const CvMat* sample, bool returnDFVal=false ) const;
    virtual float predict( const CvMat* samples, CV_OUT CvMat* results ) const;

    CV_WRAP DorniSVM( const cv::Mat& trainData, const cv::Mat& responses,
          const cv::Mat& varIdx=cv::Mat(), const cv::Mat& sampleIdx=cv::Mat(),
          CvSVMParams params=CvSVMParams() );

    CV_WRAP virtual bool train( const cv::Mat& trainData, const cv::Mat& responses,
                       const cv::Mat& varIdx=cv::Mat(), const cv::Mat& sampleIdx=cv::Mat(),
                       CvSVMParams params=CvSVMParams() );

    CV_WRAP virtual bool train_auto( const cv::Mat& trainData, const cv::Mat& responses,
                            const cv::Mat& varIdx, const cv::Mat& sampleIdx, CvSVMParams params,
                            int k_fold = 10,
                            CvParamGrid Cgrid      = DorniSVM::get_default_grid(DorniSVM::C),
                            CvParamGrid gammaGrid  = DorniSVM::get_default_grid(DorniSVM::GAMMA),
                            CvParamGrid pGrid      = DorniSVM::get_default_grid(DorniSVM::P),
                            CvParamGrid nuGrid     = DorniSVM::get_default_grid(DorniSVM::NU),
                            CvParamGrid coeffGrid  = DorniSVM::get_default_grid(DorniSVM::COEF),
                            CvParamGrid degreeGrid = DorniSVM::get_default_grid(DorniSVM::DEGREE),
                            bool balanced=false);
    CV_WRAP virtual float predict( const cv::Mat& sample, bool returnDFVal=false ) const;
    CV_WRAP_AS(predict_all) void predict( cv::InputArray samples, cv::OutputArray results ) const;
    
    CV_WRAP virtual float decision_func_score( const CvMat* sample ) const;
    CV_WRAP virtual int get_support_vector_count() const;
    virtual const float* get_support_vector(int i) const;
    virtual CvSVMParams get_params() const { return params; };
    CV_WRAP virtual void clear();

    static CvParamGrid get_default_grid( int param_id );

    virtual void write( CvFileStorage* storage, const char* name ) const;
    virtual void read( CvFileStorage* storage, CvFileNode* node );
    CV_WRAP int get_var_count() const { return var_idx ? var_idx->cols : var_all; }

protected:

    virtual bool set_params( const CvSVMParams& params );
    virtual bool train1( int sample_count, int var_count, const float** samples,
                    const void* responses, double Cp, double Cn,
                    CvMemStorage* _storage, double* alpha, double& rho );
    virtual bool do_train( int svm_type, int sample_count, int var_count, const float** samples,
                    const CvMat* responses, CvMemStorage* _storage, double* alpha );
    virtual void create_kernel();
    virtual void create_solver();

    virtual float predict( const float* row_sample, int row_len, bool returnDFVal=false ) const;

    virtual void write_params( CvFileStorage* fs ) const;
    virtual void read_params( CvFileStorage* fs, CvFileNode* node );

    void optimize_linear_svm();

    CvSVMParams params;
    CvMat* class_labels;
    int var_all;
    float** sv;
    int sv_total;
    CvMat* var_idx;
    CvMat* class_weights;
    CvSVMDecisionFunc* decision_func;
    CvMemStorage* storage;

    CvSVMSolver* solver;
    CvSVMKernel* kernel;

private:
    DorniSVM(const DorniSVM&);
    DorniSVM& operator = (const DorniSVM&);
    
};