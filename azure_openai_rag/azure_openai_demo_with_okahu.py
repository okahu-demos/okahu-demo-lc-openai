import sys
import auzure_openai_llm_demo
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from okahu_apptrace.wrap_common import llm_wrapper
from okahu_apptrace.wrapper import WrapperMethod
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from credential_utilties.environment import setOkahuEnvironmentVariablesFromConfig

def main():
    #Okahu instrumentation
    setOkahuEnvironmentVariablesFromConfig(sys.argv[1])
    setup_okahu_telemetry(workflow_name="azure_openai_llama_index_2")

    #invoke the underlying application
    auzure_openai_llm_demo.main()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage " + sys.argv[0] + " <config-file-path>")
        sys.exit(1)
    main()
