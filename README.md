# EPSpec

## Overview

This repository provides the complete implementation of all core algorithms of EPSpec. It also contains all datasets used in this project and their preprocessed versions, together with the related images, SVG figures, and experimental result figures.

## Environment and API Configuration

Some scripts in this repository rely on external large language model APIs. Before running the code, please replace the placeholder `url` / `base_url` and `api_key` fields in the relevant files with your own settings.

Examples of the OpenAI-compatible configurations used in the paper are listed below:

* **OpenAI**

  * model: `gpt-5.2`
  * base_url: `https://api.openai.com/v1`
  * api_key: `your_key`

* **GLM**

  * model: `glm-4.7`
  * base_url: `https://api.z.ai/api/paas/v4/`
  * api_key: `your_key`

* **DeepSeek**

  * model: `deepseek-chat` or `deepseek-reasoner`
  * base_url: `https://api.deepseek.com`
  * api_key: `your_key`

As large language models continue to evolve, differences in model capabilities and response behavior may affect the generated outputs. Therefore, this repository also provides a validation script for checking whether model-generated JSON responses conform to the expected structure and can be parsed correctly.

Readers may also experiment with newer or more advanced models. However, the output format and content consistency should be carefully verified before incorporating a new model into the workflow.

Some file paths used in the code are environment-specific. Please replace them with the corresponding paths on your local machine or in your current runtime environment.

After completing the required configuration, you may execute the relevant experimental scripts in their corresponding directories.

## Prompt Configuration

For the prompt-related components of this repository, readers are advised to use the original Chinese prompts by default, as these are the versions used in the experiments reported in the paper.

English translations may be prepared when necessary for readability, adaptation, or reproduction. However, the translated prompts should be carefully verified to ensure that their meanings and instructions remain consistent with those of the original Chinese prompts.

## Parameter Settings

Most parameters can be used with their default settings. For PLSR, the number of latent variables is selected using the one-standard-error (one-SE) rule. The default upper bound for the number of latent variables is 30, which can be used for the shootout and soil datasets. For the corn dataset, an upper bound of 10 is recommended because of the behavior of its validation-error curve as the number of latent variables increases.

The selection and analysis of some parameter settings are discussed in the paper. For parameters not explicitly specified, readers may explore alternative settings according to their own datasets and tasks or contact the authors for further information.

## Extensibility

Although the band-selection algorithm presented in this project was evaluated only for regression modeling using partial least squares regression (PLSR), the algorithm itself is not inherently tied to PLSR or to a specific predictive model.

In principle, it may also be integrated with other regression methods, such as support vector regression (SVR), principal component regression (PCR), and random forest regression. After appropriately adapting the evaluation criteria and optimization objectives, it may also be explored for classification tasks using methods such as linear discriminant analysis (LDA), support vector machines (SVM), and random forests.

To control token costs, limit the experimental scope, and keep the presentation of the paper focused and manageable, the paper reports only the most representative and commonly used setting: a near-infrared spectral regression task based on PLSR. Readers may further evaluate, adapt, and improve the proposed band-selection algorithm for other regression models, classification tasks, and application scenarios.

## Agent Framework

This repository also provides a simple example based on an agent-oriented framework. The example is intended solely to demonstrate how the relevant components can be organized and invoked within an automated analysis workflow.

Readers may adapt and extend this example according to the requirements of their own applications and experiments to construct customized workflows for automated near-infrared spectral analysis, wavelength selection, and regression modeling.

## Citation

When using this repository in academic work, please cite the corresponding paper:

```bibtex
@article{your_citation_key,
  title   = {Paper Title},
  author  = {Author Names},
  journal = {Journal Name},
  year    = {Year},
  volume  = {Volume},
  number  = {Issue},
  pages   = {Page Range},
  doi     = {DOI}
}
```

## License

The licensing terms for the code and other materials in this repository are provided in the `LICENSE` file.
