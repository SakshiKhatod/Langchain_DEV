# https://chunkviz.up.railway.app/

from langchain.text_splitter import CharacterTextSplitter

text = """APB (Advanced Peripheral Bus) is a simple, low-power, and low-cost bus protocol used for interconnecting various types of peripherals with the processor in a SoC (System on Chip) design. It is a
widely-used bus protocol in many ARM-based SoC designs.We will try to create a basic version of this protocol to understand its working. 
APB protocol defines a transaction-based communication model between the APB master and APB slave, where each transaction consists of an address phase, a data phase, and a response phase. Here's a brief explanation of each phase:
Address phase: In this phase, the APB master sends the address of the slave register to be accessed. The master also sends the read or write command to specify whether the transaction is a read or write operation. The address is sent on the paddr bus, and the read or write command is sent on the pwrite and pread signals.
Data phase: In this phase, the APB master sends the data to be written to the slave or receives the data read from the slave. The data is sent on the pwdata and prdata buses, respectively.
Response phase: In this phase, the APB slave sends a response signal to indicate whether the transaction was successful or not. The response signal is sent on the pready input pin. If the transaction was successful, the APB slave asserts the pready signal, and the APB master can proceed with the next transaction. If the transaction was unsuccessful, the APB slave de-asserts the pready signal, and the APB master must retry the transaction.
Also, check here for easy explanation of APB protocol."""

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
)

result = splitter.split_text(text)
print(result)
