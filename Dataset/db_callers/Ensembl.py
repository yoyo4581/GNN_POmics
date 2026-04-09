from mygene import MyGeneInfo

def EnsemblCaller(EnsemblIds):
    mg = MyGeneInfo()
    EnsemblIds = [i.split('.')[0] for i in EnsemblIds]

    result = mg.querymany(
        EnsemblIds, 
        scopes='ensembl.gene', 
        fields='all', 
        species='human'
    )

    results_map = {EnsId: None for EnsId in EnsemblIds}
    second_map = {}
    for item in result:
        ensemble_query = item['query']
        if 'entrezgene' in item:
            results_map[ensemble_query] = item['entrezgene']
        elif 'symbol' in item:
            results_map[ensemble_query] = item['symbol']
            second_map[item['symbol']] = ensemble_query
        
        
    if second_map:
        result2 = mg.querymany(
            list(second_map.keys()),
            scopes='symbol',
            fields='entrezgene',
            species='human'
        )
        for result in result2:
            symbol = result['query']
            
            if 'entrezgene' in result:
                ensemble_id = second_map[symbol]
                results_map[ensemble_id] = result['entrezgene']
            
    return [results_map.get(ensembleId) for ensembleId in EnsemblIds]