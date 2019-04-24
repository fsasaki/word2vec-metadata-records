def sparql_query():
    import requests
    url = "http://linkeddata.econstor.eu/beta/sparql"
    query = '''
SELECT ?publication ?title (GROUP_CONCAT(?keyword;SEPARATOR="XXXYYY") as ?keywords) WHERE {
  ?publication <http://purl.org/dc/elements/1.1/title> ?title.
  ?publication <http://purl.org/dc/elements/1.1/keyword> ?keyword.
}
GROUP BY ?publication ?title
'''
    params = {"query": query, "format":"text/csv"}
    headers = {'Accept': "text/csv"}
    response = requests.request("GET", url, headers=headers, params=params)
    f = open('result-with-delimiter.csv', 'w',encoding="utf-8")
    f.write(response.text)
    f.close()