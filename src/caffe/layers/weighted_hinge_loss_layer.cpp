#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  // Copy bottom activation to bottom differentiation
  caffe_copy(count, bottom_data, bottom_diff);

  for (int i = 0; i < num; ++i) {
    //==============================================================
    // Cache corect_activation for this sample
    Dtype correct_activation = bottom_diff[i * dim + static_cast<int>(label[i])];
    bottom_diff[i * dim + static_cast<int>(label[i])] = 0;
    
    for (int j = 0; j < dim; ++j) {
      if( j == static_cast<int>(label[i]) ) continue;
      // Pairwise ranking weight
      Dtype weight = (std::pow(static_cast<int>(label[i]) - j, 2) / std::pow(dim, 2) );
      // Dtype weight = 1; 
      
      // Pairwise margin for each 
      Dtype margin = 1 + bottom_diff[i * dim + j] - correct_activation;

      bottom_diff[i * dim + j] = std::max(Dtype(0), margin) * weight;
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
}

template <typename Dtype>
void WeightedHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    // Copy bottom activation to bottom differentiation
    caffe_copy(count, bottom_data, bottom_diff);

    for (int i = 0; i < num; ++i) {
      //==============================================================
      // Cache corect_activation for this sample
      Dtype correct_activation = bottom_diff[i * dim + static_cast<int>(label[i])];
      bottom_diff[i * dim + static_cast<int>(label[i])] = 0;

      for (int j = 0; j < dim; ++j) {
        if( j == static_cast<int>(label[i]) ) continue;
        //==============================================================
        // Pairwise ranking weight
        Dtype weight = (std::pow(static_cast<int>(label[i]) - j, 2) / std::pow(dim, 2) );
        // Dtype weight = 1; 
	      // LOG(INFO) << "weight on back propagate phase: " << weight << "\n";

        //==============================================================
        // Pairwise margin for each
        Dtype margin = 1 + bottom_diff[i * dim + j] - correct_activation;

        if( margin > Dtype(0) )
        {
	      // LOG(INFO) << "bottom_diff[" << (i * dim + j) << "] = "<< weight << "\n";
	      // LOG(INFO) << "bottom_diff[" << (i * dim + static_cast<int>(label[i])) << "] = " 
	      //	<< ( bottom_diff[i * dim +  static_cast<int>(label[i])] - weight) << "\n";
          bottom_diff[i * dim + j] = weight;
          bottom_diff[i * dim +  static_cast<int>(label[i])] -= weight;
        }
        else
          bottom_diff[i * dim + j] = Dtype(0);

      }
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(WeightedHingeLossLayer);
REGISTER_LAYER_CLASS(WeightedHingeLoss);

} // namespace caffe
