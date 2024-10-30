from docquery import document, pipeline
p = pipeline('document-question-answering')
doc = document.load_document("sample-invoice-template.png")
q = "What is the invoice number?"
print(q, p(question=q, **doc.context))
