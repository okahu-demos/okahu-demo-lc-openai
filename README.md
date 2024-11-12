# Okahu demo with OpenAI + Langchain
This repo includes a demo chat application built using OpenAI & Langchain that is pre-instrumented for observation with Okahu AI Observability cloud. 
You can fork this repo and run the app in Github Codespaces to get started quickly. 


## Try Okahu with this OpenAI app

To try this chatbot 
- Fork this repo and run in the Github Codespace 
- To run this on Mac 
  - Install python dependencies: ```pip install -r requirement.txt```
  - Remove libmagic: ```pip uninstall python-magic```
- To run this on Windows
  - Install python dependencies: ```pip install -r requirement.txt```

You'll need 
- An OpenAI subscription and an API key to [OpenAI developer platform](https://platform.openai.com/overview)
- An Okahu tenant and API key to [Okahu AI Observability Cloud](https://www.okahu.ai)  

## Configure the demo environment
- Copy the file botenv.sh.template to botenv.sh
- Edit the botenv.sh file to add the OpenAI API Key and Okahu API key and save

## Run the interactive chatbot 
This application is an interactive chatbot that answers questions about coffee and built with a RAG design pattern.
Workflow is a python program using Langchain LLM orchestration framework. 
The vector dataset is built using text-embedding-3-large embedding model from OpenAI from a [local dataset about coffee](data/coffee.txt). The vector data is stored in a local filebased Chroma vectorDB. 
The app uses OpenAI gpt-4o-mini model for inference.

To try Okahu from the Github Codespace 

1. Run the pre-instrumented chatbot app with following command from top level directory

   ```python lc-openai-with-okahu.py```
   
2. View the workflow discovered by Okahu AI Observability Cloud with following commands with your Okahu API key
    - Discover all components
      ```curl --location --request PUT 'https://api.okahu.ai/api/v1/discovery' --header 'x-api-key: <YOUR_OKAHU_API_KEY>;' ```
    - Get discovered components
      ```curl --location 'https://api.okahu.ai/api/v1/components' --header 'x-api-key: <YOUR_OKAHU_API_KEY>;' ```

    Check out Okahu AI Observability Cloud API docs [here](https://apidocs.okahu.ai)

### Example output 

```
$ python lc-openai-with-okahu.py 

Ask a coffee question [Press return to exit]: What is an americano?
An americano is a type of coffee drink that is made by diluting an espresso shot with hot water at a 1:3 to 1:4 ratio, resulting in a drink that retains the complex flavors of espresso, but in a lighter way.
```

### Okahu instrumentation

To run the chatbot app without Okahu instrumentation, use the command ```python lc-openai.py```

To understand how Okahu instrumentation works, compare the [lc-openai.py](lc-openai.py) and [lc-openai-with-okahu.py](lc-openai-with-okahu.py)
