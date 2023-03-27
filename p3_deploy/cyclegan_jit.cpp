/**
 * @file sample_cyclegan_jit.cpp
 * @author Shinichi Inoue (inoue.shinichi.1800@gmail.com)
 * @brief C++からJIT化したモデルの呼び出し
 * @version 0.1
 * @date 2023-03-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "torch/script.h"

#define cimg_use_jpeg;

#include "CImg.h"

using namespace cimg_library;

#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Call as " << argv[0] 
            << " model.pt input.jpg output.jpg"
            << std::endl;
        return 1;
    }

    CImg<float> image(argv[2]);
    image = image.resize(227, 227);

    auto input_ = torch::tensor(
        torch::ArrayRef<float>(image.data(), image.size())
    );

    auto input = input_.reshape({1, 3, image.height(), image.width()}).div_(255); // [0,1]正規化

    auto module = torch::jit::load(argv[1]);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    
    auto output_ = module.forward(inputs).toTensor();
    auto output = output_.congiguous().mul_(255);

    CImg<float> out_img(output.data_ptr<float>(), 
                        output.size(2),
                        output.size(3),
                        1,
                        output.size(1));
    out_img.save(argv[3]);
    return 0;
}