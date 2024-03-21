from fastapi.testclient import TestClient
from gis import app

#Webアプリケーションが起動していなくてもTestClientによるテストは実行できる。
test_client = TestClient(app)

def test_get_hellogis():
    response = test_client.get("/hellogis/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Brest GIS!"}

def test_get_point_side_of_line():
    response = test_client.post("/pointsideofline/",
                                json={"point":{"type":"Feature","properties":{},"geometry":{"coordinates":[139.751363304702,35.65771000179585],"type":"Point"}},"line":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.7551995346509,35.66006748781244],[139.75693265376594,35.65465624195437]],"type":"LineString"}}]}})
    assert response.status_code == 200
    assert response.json() == {"side": -1}

def test_check_cross_lines():
    response = test_client.post("linecrosscheck",
                                json={"line1":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75644916016995,35.6580874134024],[139.75590014523328,35.655063923545256]],"type":"LineString"}}]},"line2":{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75800034522894,35.657506798937305],[139.7541398116271,35.6560694059431]],"type":"LineString"}}})
    assert response.status_code == 200
    assert response.json() == {"result": True}

def test_calc_convexhull():
    response = test_client.post("convexhull", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75463668298983,35.65737926565541],[139.75712903651032,35.65761292845477],[139.75978130764446,35.65497571395987],[139.75199640429526,35.655462293990055],[139.75199640429526,35.655462293990055],[139.75953804242914,35.65869735639879],[139.75760608843967,35.66130862409193]],"type":"MultiPoint"}}]})
    assert response.status_code == 200
    assert response.json() == {"result":{"type":"Polygon","coordinates":[[[139.75978130764446,35.65497571395987],[139.75199640429526,35.655462293990055],[139.75760608843967,35.66130862409193],[139.75953804242914,35.65869735639879],[139.75978130764446,35.65497571395987]]]}}
    
def test_calc_trianglation():
    response = test_client.post("triangulation", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[139.7561455945392,35.65964771465745],[139.75268156823682,35.65887169603326],[139.75436368800598,35.655755963250684],[139.7588255819664,35.655744380374244],[139.75881132671503,35.65778294077418],[139.75620261554775,35.65808408733078],[139.7561455945392,35.65964771465745]]],"type":"Polygon"}}]})
    assert response.status_code == 200
    assert response.json() == {"result":[{"type":"Polygon","coordinates":[[[139.75268156823682,35.65887169603326],[139.75436368800598,35.655755963250684],[139.75620261554775,35.65808408733078],[139.75268156823682,35.65887169603326]]]},{"type":"Polygon","coordinates":[[[139.75268156823682,35.65887169603326],[139.75620261554775,35.65808408733078],[139.7561455945392,35.65964771465745],[139.75268156823682,35.65887169603326]]]},{"type":"Polygon","coordinates":[[[139.7561455945392,35.65964771465745],[139.75620261554775,35.65808408733078],[139.75881132671503,35.65778294077418],[139.7561455945392,35.65964771465745]]]},{"type":"Polygon","coordinates":[[[139.75881132671503,35.65778294077418],[139.75620261554775,35.65808408733078],[139.7588255819664,35.655744380374244],[139.75881132671503,35.65778294077418]]]},{"type":"Polygon","coordinates":[[[139.7588255819664,35.655744380374244],[139.75620261554775,35.65808408733078],[139.75436368800598,35.655755963250684],[139.7588255819664,35.655744380374244]]]}]}

def test_calc_minimumboundingcircle():
    response = test_client.post("minimumboundingcircle", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75374449360527,35.658523066803696],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.76201841156382,35.65988717456693],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7567525507992,35.65414159517911],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.76088452873603,35.66485411418411],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.74956361404787,35.65265046177974],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.730688201693,35.668915848160225],"type":"Point"}}]})
    assert response.status_code == 200
    assert response.json() == {"result":{"type":"Polygon","coordinates":[[[139.7648517968404,35.66002469272802],[139.7644850412629,35.65630096085935],[139.76339886874166,35.65272032991809],[139.76163502027748,35.64942040154329],[139.7592612795777,35.64652799013176],[139.75636886816616,35.64415424943196],[139.75306893979135,35.64239040096779],[139.74948830885012,35.64130422844654],[139.74576457698143,35.64093747286907],[139.74204084511274,35.64130422844654],[139.7384602141715,35.64239040096779],[139.7351602857967,35.64415424943196],[139.73226787438517,35.64652799013176],[139.72989413368538,35.64942040154329],[139.7281302852212,35.65272032991809],[139.72704411269996,35.65630096085935],[139.72667735712247,35.66002469272802],[139.72704411269996,35.6637484245967],[139.7281302852212,35.667329055537955],[139.72989413368538,35.670628983912756],[139.73226787438517,35.673521395324286],[139.7351602857967,35.67589513602408],[139.7384602141715,35.677658984488254],[139.74204084511274,35.6787451570095],[139.74576457698143,35.679111912586976],[139.74948830885012,35.6787451570095],[139.75306893979135,35.677658984488254],[139.75636886816616,35.67589513602408],[139.7592612795777,35.673521395324286],[139.76163502027748,35.670628983912756],[139.76339886874166,35.667329055537955],[139.7644850412629,35.6637484245967],[139.7648517968404,35.66002469272802]]]}}    

def test_contains():
    response = test_client.post("contains", 
                                json={"area_geojson":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[139.75440423823215,35.65489368456221],[139.753023928243,35.652420204219695],[139.7548848818882,35.651188442524116],[139.75942018613893,35.65092806770086],[139.75964202167364,35.65245024695034],[139.75806452454225,35.65466336371588],[139.75440423823215,35.65489368456221]],[[139.75717718240605,35.6536018764493],[139.75726345178168,35.65234009021505],[139.75589546598803,35.652400175725816],[139.75594476277286,35.6535417918425],[139.75717718240605,35.6536018764493]]],"type":"Polygon"}}]},"target_geojson":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75306090083097,35.65465334975107],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75638843384098,35.65527421320071],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7570785888363,35.650417330006434],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7581138213285,35.65163908925017],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75490953028208,35.65376210184603],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75641308223345,35.65328142469099],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75699231946209,35.652760687841194],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75506974483358,35.65184939018542],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75320879118703,35.651388730271734],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7589518666784,35.6534616789628],"type":"Point"}}]}})
    assert response.status_code == 200
    assert response.json() == {"result":{"0":[{"type":"Point","coordinates":[139.7581138213285,35.65163908925017]},{"type":"Point","coordinates":[139.75490953028208,35.65376210184603]},{"type":"Point","coordinates":[139.75506974483358,35.65184939018542]}]}}
    