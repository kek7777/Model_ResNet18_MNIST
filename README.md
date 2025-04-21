<img src="objects\Resnet.gif" width="800" height="150">

<a id="readme-top"></a>

<!-- TITLE -->
#  **ResNet18 using Pytorch**

This project created to looking into ML model to solve a 10 image classification problem.
You can analyze the accuracy of a model based on its hyperparameters.

<!-- TABLE OF CONTENTS -->
<details>
  <summary> Table of Contents </summary>
  <ol>
    <li><a href="#Model's structure">Model's structure</a></li>
    <li>
          <a href="#getting-started">Getting Started</a>
          <ul>
            <li><a href="#file assignment">File assignment</a></li>
            <li><a href="#installation">Installation</a></li>
          </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#Source of information">Source of information</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- MODEL's STRUCTURE -->
## Model's structure

 <img src="objects\arh_resnet18.png" width="800" height="200">

**The model includes:**
* Input layer (224x224x1)
* Convolution layer ( kernel - 7x7, channels - 64, stride - 2, padding - 3)
* Subsampling layer (max pooling 3x3, stride - 2, padding - 1)
* Residual block groups - 4:  
group 1:  2 residual block (kernel - 3x3, channels - 64, stride - 1)  
group 2:  2 residual block (kernel - 3x3, channels - 128, stride - 2)  
group 3:  2 residual block (kernel - 3x3, channels - 256, stride - 2)  
group 4:  2 residual block (kernel - 3x3, channels - 512, stride - 2)  
* Subsampling layer (average pooling 1x1)
* Fully connected layer (channels - 10)

Model's activation functions are RELU. 

 
<!-- GETTING STARTED -->
## Getting Started

The information about file assignment of project and how that used.

### File assignment

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```
 <p align="right">(<a href="#readme-top">back to top</a>)</p>


 <!-- LICENSE -->
 ## License
 
 Distributed under the Unlicense License. See `LICENSE.txt` for more information.
 
 <p align="right">(<a href="#readme-top">back to top</a>)</p>
 
 <!-- Source of information -->
 ## Source of information
 
 Information that was used in the development of the project.
 
 * [Origin article of ResNet](https://arxiv.org/abs/1512.03385)
 * [ResNet using PyTorch](https://pytorch.org/hub/pytorch_vision_resnet/)
 * [Example of the ResNet18](https://www.kaggle.com/code/ivankunyankin/resnet18-from-scratch-using-pytorch)
 
 <p align="right">(<a href="#readme-top">back to top</a>)</p>

  <!-- CONTACT -->
 ## Contact
 
 Author of the project - [Email](https://kek777771.gmail.com)
 
 Project Link: [https://github.com](https://github.com/kek7777/Model_ResNet18_MNIST.git)
 
 <p align="right">(<a href="#readme-top">back to top</a>)</p>
 
 
