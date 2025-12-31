from fastapi.testclient import TestClient
from gis import app
from pytest import approx, fixture

#Webアプリケーションが起動していなくてもTestClientによるテストは実行できる。
test_client = TestClient(app)

def assert_equals_json_values(actual: dict, expected: dict, tolerance: float = 1.2e-13):
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in actual:
            if key in expected:
                if isinstance(actual[key], (int, float)) and isinstance(expected[key], (int, float)):
                    assert actual[key] == approx(expected[key], abs=tolerance)
                elif isinstance(actual[key], dict) and isinstance(expected[key], dict):
                    assert_equals_json_values(actual[key], expected[key])
                elif isinstance(actual[key], list) and isinstance(expected[key], list):
                    assert len(actual[key]) == len(expected[key]), "Lists are of different lengths"
                    for act_item, exp_item in zip(actual[key], expected[key]):
                        assert_equals_json_values(act_item, exp_item)  # リストの要素を再帰的に比較
                else:
                    assert actual[key] == expected[key]
            else:
                raise AssertionError(f"Key {key} not found in actual JSON object")
        for key in expected:  # actual にないキーが expected にある場合のエラーチェック
            if key not in actual:
                raise AssertionError(f"Key {key} not found in actual JSON object")
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), "Lists are of different lengths"
        for item_a, item_b in zip(actual, expected):
            assert_equals_json_values(item_a, item_b)
    else:
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            assert actual == approx(expected, abs=tolerance)
        else:
            assert actual == expected, "Values do not match"

def test_get_hellogis():
    response = test_client.get("/hellogis/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Brest GIS!"}

def test_get_point_side_of_line():
    response = test_client.post("/pointsideofline/",
                                json={"point":{"type":"Feature","properties":{},"geometry":{"coordinates":[139.751363304702,35.65771000179585],"type":"Point"}},"line":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.7551995346509,35.66006748781244],[139.75693265376594,35.65465624195437]],"type":"LineString"}}]}})
    assert response.status_code == 200
    result = response.json()
    assert_equals_json_values(result, {"side": -1})

def test_check_cross_lines():
    response = test_client.post("linecrosscheck",
                                json={"line1":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75644916016995,35.6580874134024],[139.75590014523328,35.655063923545256]],"type":"LineString"}}]},"line2":{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75800034522894,35.657506798937305],[139.7541398116271,35.6560694059431]],"type":"LineString"}}})
    assert response.status_code == 200
    assert response.json() == {"result": True}

def test_calc_convexhull():
    response = test_client.post("convexhull", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75463668298983,35.65737926565541],[139.75712903651032,35.65761292845477],[139.75978130764446,35.65497571395987],[139.75199640429526,35.655462293990055],[139.75199640429526,35.655462293990055],[139.75953804242914,35.65869735639879],[139.75760608843967,35.66130862409193]],"type":"MultiPoint"}}]})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"type":"Polygon","coordinates":[[[139.75978130764446,35.65497571395987],[139.75199640429526,35.655462293990055],[139.75760608843967,35.66130862409193],[139.75953804242914,35.65869735639879],[139.75978130764446,35.65497571395987]]]}})
    
def test_calc_trianglation():
    response = test_client.post("triangulation", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[139.7561455945392,35.65964771465745],[139.75268156823682,35.65887169603326],[139.75436368800598,35.655755963250684],[139.7588255819664,35.655744380374244],[139.75881132671503,35.65778294077418],[139.75620261554775,35.65808408733078],[139.7561455945392,35.65964771465745]]],"type":"Polygon"}}]})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":[{"type":"Polygon","coordinates":[[[139.75268156823682,35.65887169603326],[139.75436368800598,35.655755963250684],[139.75620261554775,35.65808408733078],[139.75268156823682,35.65887169603326]]]},{"type":"Polygon","coordinates":[[[139.75268156823682,35.65887169603326],[139.75620261554775,35.65808408733078],[139.7561455945392,35.65964771465745],[139.75268156823682,35.65887169603326]]]},{"type":"Polygon","coordinates":[[[139.7561455945392,35.65964771465745],[139.75620261554775,35.65808408733078],[139.75881132671503,35.65778294077418],[139.7561455945392,35.65964771465745]]]},{"type":"Polygon","coordinates":[[[139.75881132671503,35.65778294077418],[139.75620261554775,35.65808408733078],[139.7588255819664,35.655744380374244],[139.75881132671503,35.65778294077418]]]},{"type":"Polygon","coordinates":[[[139.7588255819664,35.655744380374244],[139.75620261554775,35.65808408733078],[139.75436368800598,35.655755963250684],[139.7588255819664,35.655744380374244]]]}]})

def test_calc_minimumboundingcircle():
    response = test_client.post("minimumboundingcircle", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75374449360527,35.658523066803696],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.76201841156382,35.65988717456693],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7567525507992,35.65414159517911],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.76088452873603,35.66485411418411],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.74956361404787,35.65265046177974],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.730688201693,35.668915848160225],"type":"Point"}}]})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"type":"Polygon","coordinates":[[[139.7648517968404,35.66002469272802],[139.7644850412629,35.65630096085935],[139.76339886874166,35.65272032991809],[139.76163502027748,35.64942040154329],[139.7592612795777,35.64652799013176],[139.75636886816616,35.64415424943196],[139.75306893979135,35.64239040096779],[139.74948830885012,35.64130422844654],[139.74576457698143,35.64093747286907],[139.74204084511274,35.64130422844654],[139.7384602141715,35.64239040096779],[139.7351602857967,35.64415424943196],[139.73226787438517,35.64652799013176],[139.72989413368538,35.64942040154329],[139.7281302852212,35.65272032991809],[139.72704411269996,35.65630096085935],[139.72667735712247,35.66002469272802],[139.72704411269996,35.6637484245967],[139.7281302852212,35.667329055537955],[139.72989413368538,35.670628983912756],[139.73226787438517,35.673521395324286],[139.7351602857967,35.67589513602408],[139.7384602141715,35.677658984488254],[139.74204084511274,35.6787451570095],[139.74576457698143,35.679111912586976],[139.74948830885012,35.6787451570095],[139.75306893979135,35.677658984488254],[139.75636886816616,35.67589513602408],[139.7592612795777,35.673521395324286],[139.76163502027748,35.670628983912756],[139.76339886874166,35.667329055537955],[139.7644850412629,35.6637484245967],[139.7648517968404,35.66002469272802]]]}})    

def test_contains():
    response = test_client.post("contains", 
                                json={"area_geojson":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[139.75440423823215,35.65489368456221],[139.753023928243,35.652420204219695],[139.7548848818882,35.651188442524116],[139.75942018613893,35.65092806770086],[139.75964202167364,35.65245024695034],[139.75806452454225,35.65466336371588],[139.75440423823215,35.65489368456221]],[[139.75717718240605,35.6536018764493],[139.75726345178168,35.65234009021505],[139.75589546598803,35.652400175725816],[139.75594476277286,35.6535417918425],[139.75717718240605,35.6536018764493]]],"type":"Polygon"}}]},"target_geojson":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75306090083097,35.65465334975107],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75638843384098,35.65527421320071],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7570785888363,35.650417330006434],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7581138213285,35.65163908925017],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75490953028208,35.65376210184603],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75641308223345,35.65328142469099],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75699231946209,35.652760687841194],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75506974483358,35.65184939018542],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75320879118703,35.651388730271734],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7589518666784,35.6534616789628],"type":"Point"}}]}})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"0":[{"type":"Point","coordinates":[139.7581138213285,35.65163908925017]},{"type":"Point","coordinates":[139.75490953028208,35.65376210184603]},{"type":"Point","coordinates":[139.75506974483358,35.65184939018542]}]}})

def test_calc_voronoi_diagram():
    response = test_client.post("voronoidiagram", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7272138350341,35.65889531206699],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75324475259072,35.66104334746616],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.74368652505035,35.65410333257785],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75268549459713,35.64592325509041],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7457710321213,35.647162714612975],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.73351819788098,35.64633641040095],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.72650205213301,35.653442347325395],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7344333473252,35.65021996588669],"type":"Point"}}]})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"type":"GeometryCollection","geometries":[{"type":"Polygon","coordinates":[[[139.6997593516753,35.687786047923865],[139.7379338966117,35.687786047923865],[139.73939627569436,35.670064240661425],[139.73618270820987,35.65901742772361],[139.73195948837878,35.65550291811847],[139.6997593516753,35.65970604573877],[139.6997593516753,35.687786047923865]]]},{"type":"Polygon","coordinates":[[[139.6997593516753,35.65970604573877],[139.73195948837878,35.65550291811847],[139.7294568214346,35.649343066878124],[139.6997593516753,35.62002085667664],[139.6997593516753,35.65970604573877]]]},{"type":"Polygon","coordinates":[[[139.7294568214346,35.649343066878124],[139.73195948837878,35.65550291811847],[139.73618270820987,35.65901742772361],[139.74026420240583,35.64929215731188],[139.7396314045341,35.64694545363357],[139.7294568214346,35.649343066878124]]]},{"type":"Polygon","coordinates":[[[139.77998745304842,35.619180554632706],[139.7443233810471,35.619180554632706],[139.75025916602817,35.65229398977839],[139.75139310628006,35.65354144664056],[139.77998745304842,35.65248380647513],[139.77998745304842,35.619180554632706]]]},{"type":"Polygon","coordinates":[[[139.6997593516753,35.619180554632706],[139.6997593516753,35.62002085667664],[139.7294568214346,35.649343066878124],[139.7396314045341,35.64694545363357],[139.74150380830133,35.619180554632706],[139.6997593516753,35.619180554632706]]]},{"type":"Polygon","coordinates":[[[139.73618270820987,35.65901742772361],[139.73939627569436,35.670064240661425],[139.75139310628006,35.65354144664056],[139.75025916602817,35.65229398977839],[139.74026420240583,35.64929215731188],[139.73618270820987,35.65901742772361]]]},{"type":"Polygon","coordinates":[[[139.74150380830133,35.619180554632706],[139.7396314045341,35.64694545363357],[139.74026420240583,35.64929215731188],[139.75025916602817,35.65229398977839],[139.7443233810471,35.619180554632706],[139.74150380830133,35.619180554632706]]]},{"type":"Polygon","coordinates":[[[139.77998745304842,35.687786047923865],[139.77998745304842,35.65248380647513],[139.75139310628006,35.65354144664056],[139.73939627569436,35.670064240661425],[139.7379338966117,35.687786047923865],[139.77998745304842,35.687786047923865]]]}]}})

def test_split_polygon_by_line():
    response = test_client.post("splitpolygon", 
                                json={"polygon":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[139.75232572040687,35.65724406299894],[139.75102264635348,35.653367496833766],[139.7595136450251,35.65246236838786],[139.75976585290647,35.65736360169613],[139.75232572040687,35.65724406299894]]],"type":"Polygon"}}]},"line":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.7532925172859,35.65813206047791],[139.75854684814607,35.65198418318164]],"type":"LineString"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75242694487304,35.652000337592185],[139.76211535757471,35.658182001344]],"type":"LineString"}}]}})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"type":"GeometryCollection","geometries":[{"type":"Polygon","coordinates":[[[139.75800028867047,35.65262369003427],[139.7540619009458,35.653043516593954],[139.75637831338994,35.654521496865954],[139.75800028867047,35.65262369003427]]]},{"type":"Polygon","coordinates":[[[139.7540619009458,35.653043516593954],[139.75102264635348,35.653367496833766],[139.75232572040687,35.65724406299894],[139.7540280751989,35.65727141429598],[139.75637831338994,35.654521496865954],[139.7540619009458,35.653043516593954]]]},{"type":"Polygon","coordinates":[[[139.7540280751989,35.65727141429598],[139.75976585290647,35.65736360169613],[139.75972963659402,35.65665979896035],[139.75637831338994,35.654521496865954],[139.7540280751989,35.65727141429598]]]},{"type":"Polygon","coordinates":[[[139.75972963659402,35.65665979896035],[139.7595136450251,35.65246236838786],[139.75800028867047,35.65262369003427],[139.75637831338994,35.654521496865954],[139.75972963659402,35.65665979896035]]]}]}})

def test_get_nearest_point():
    response = test_client.post("nearestpoint", 
                                json={"points":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75241455921747,35.657536211959695],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75901360821473,35.65605059515178],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75640761434295,35.65562368865875],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75609237314808,35.65328420056059],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.74997669398107,35.65118372585886],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.74821134329318,35.654394185209114],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75968612276336,35.65155942457709],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7550415691676,35.65000538659862],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7534443471174,35.653864809839334],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.76000136395584,35.65314758599911],"type":"Point"}}]},"line":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75833246324584,35.659343144469915],[139.75667219295622,35.65280295632168],[139.75066159418748,35.65367387390418],[139.74952672588694,35.651710026703626]],"type":"LineString"}}]}})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[139.75603599928445,35.65289513886054]},{"type":"Point","coordinates":[139.75609237314808,35.65328420056059]}]}})

def test_get_buffer_near_points():
    response = test_client.post("buffernearpoints", 
                                json={"points":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75241455921747,35.657536211959695],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75901360821473,35.65605059515178],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75640761434295,35.65562368865875],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75609237314808,35.65328420056059],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.74997669398107,35.65118372585886],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.74821134329318,35.654394185209114],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75968612276336,35.65155942457709],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7550415691676,35.65000538659862],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.7534443471174,35.653864809839334],"type":"Point"}},{"type":"Feature","properties":{},"geometry":{"coordinates":[139.76000136395584,35.65314758599911],"type":"Point"}}]},"line":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75833246324584,35.659343144469915],[139.75667219295622,35.65280295632168],[139.75066159418748,35.65367387390418],[139.74952672588694,35.651710026703626]],"type":"LineString"}}]},"distance":0.001})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[139.75640761434295,35.65562368865875]},{"type":"Point","coordinates":[139.75609237314808,35.65328420056059]},{"type":"Point","coordinates":[139.74997669398107,35.65118372585886]},{"type":"Point","coordinates":[139.7534443471174,35.653864809839334]}]},"bufferedLine":{"type":"Polygon","coordinates":[[[139.75764144952672,35.652556904090375],[139.75756911117026,35.65236075985254],[139.75745757264463,35.65218394198821],[139.75731170878566,35.65203417839283],[139.75713789463168,35.651918014544236],[139.75694372679976,35.651840527429854],[139.75673769147264,35.65180510365502],[139.75652879350665,35.65181329142998],[139.75118852337985,35.65258708041968],[139.7503925525654,35.65120968265911],[139.75027830366778,35.65105038222412],[139.75013517205755,35.65091643155654],[139.7499686582022,35.65081297830541],[139.7497851611358,35.65074399812116],[139.74959173254774,35.65071214187357],[139.74939580579013,35.650718633780386],[139.74920491021845,35.650763224361256],[139.7490263818424,35.650844200025176],[139.74886708140744,35.650958448922786],[139.74873313073985,35.65110158053302],[139.74862967748874,35.65126809438837],[139.74856069730447,35.651451591454766],[139.7485288410569,35.651645020042835],[139.74853533296368,35.65184094680043],[139.74857992354458,35.65203184237211],[139.7486608992085,35.65221037074814],[139.74979576750903,35.6541742179487],[139.74991140315197,35.654335095054975],[139.7500564860776,35.654470017217506],[139.75022532133923,35.65457368833054],[139.7504112816387,35.65464203898717],[139.75060706746797,35.654672386216006],[139.75080499363705,35.65466353879588],[139.75592448506487,35.65392173997191],[139.75736320667534,35.65958919670122],[139.7574298330776,35.659773561453044],[139.757531147051,35.65994138552159],[139.75766325515642,35.660086219522015],[139.7578210805534,35.66020249756491],[139.75799855810047,35.6602857511503],[139.75818886743502,35.660332780889576],[139.75838469507605,35.66034177945624],[139.75857851547715,35.66031240104042],[139.75876288022897,35.66024577463816],[139.7589307042975,35.66014446066474],[139.75907553829794,35.660012352559335],[139.75919181634083,35.659854527162345],[139.75927506992622,35.659677049615276],[139.7593220996655,35.65948674028073],[139.75933109823217,35.6592909126397],[139.75930171981634,35.65909709223861],[139.75764144952672,35.652556904090375]]]}})

def test_convert_coords():
    response = test_client.post("coordconvert", 
                                json={"point":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.75853087104463,35.65343414274557],"type":"Point"}}]},"fromepsg":"4301","toepsg":"4326"})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"result":{"type":"Point","coordinates":[139.75529874550224,35.656675915967796]}})

def test_calc_distance():
    response = test_client.post("distance", 
                                json={"start":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":"浜松町駅","id":0},"geometry":{"coordinates":[139.75676229423522,35.655393628051385],"type":"Point"}}]},"goal":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":"東京タワー","id":1},"geometry":{"coordinates":[139.74538477339485,35.658643501386834],"type":"Point"}}]}})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"distance":1091.540563705988,"azimuth":-70.70712778188474,"bkw_azimuth":109.28623989884333,"line":{"type":"LineString","coordinates":[[139.75676229423522,35.655393628051385],[139.74538477339485,35.658643501386834]]}},
                              tolerance = 1e-9)

def test_route_search():
    response = test_client.post("routesearch", 
                                json={"start":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"start":"true"},"geometry":{"coordinates":[139.62228254640752,35.46537173052059],"type":"Point"}}]},"goal":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[139.64015389192105,35.44347093055431],"type":"Point"}}]},"bbox":[139.62228254640752,35.44347093055431,139.64015389192105,35.46537173052059]})
    assert response.status_code == 200
    assert_equals_json_values(response.json(), {"path":{"type":"LineString","coordinates":[[139.6231189,35.4648793],[139.6231557,35.4648851],[139.6233194,35.4647936],[139.6233444,35.4647674],[139.6233445,35.4645982],[139.6233461,35.4645069],[139.6233499,35.4641905],[139.6233331,35.4641103],[139.6232972,35.4640369],[139.6232136,35.4639363],[139.6231906,35.4639089],[139.6228684,35.4635263],[139.6227719,35.4634666],[139.6227449,35.463453],[139.6226918,35.4634263],[139.622685,35.4634179],[139.6226768,35.4634078],[139.622472,35.4631566],[139.6224364,35.4631153],[139.6226114,35.4630186],[139.6226316,35.4628795],[139.6226848,35.462513],[139.6227553,35.462112],[139.6227644,35.462061],[139.6227776,35.462003],[139.6228681,35.4617327],[139.6229534,35.4614857],[139.6230849,35.4610849],[139.6231194,35.4610215],[139.623371,35.4607079],[139.6234345,35.4606441],[139.6235034,35.4606076],[139.623565,35.4605921],[139.6236808,35.4605871],[139.624058,35.4606133],[139.6240484,35.4605125],[139.6240006,35.4602757],[139.6239029,35.4600327],[139.6238852,35.4599268],[139.6238679,35.4598744],[139.6236695,35.4595473],[139.6236609,35.4595132],[139.6236603,35.4594701],[139.6236685,35.4594407],[139.6236818,35.459409],[139.6237101,35.4593568],[139.6243039,35.4586345],[139.6252169,35.4574917],[139.6252442,35.4574585],[139.6254104,35.4572557],[139.6261123,35.4563811],[139.6262312,35.4562342],[139.6263036,35.4561449],[139.6263875,35.4560416],[139.6269553,35.4553426],[139.6273289,35.454879],[139.6274046,35.454785],[139.6276512,35.4544788],[139.6286937,35.4531788],[139.629105,35.4526784],[139.6291592,35.4526119],[139.6292226,35.4525318],[139.6292766,35.4524655],[139.6301809,35.4513226],[139.6310363,35.4502651],[139.6310848,35.4502041],[139.6311582,35.4501118],[139.6317267,35.4493965],[139.6317938,35.4493237],[139.6318735,35.449251],[139.6320535,35.4491002],[139.6323379,35.4488995],[139.632464,35.4488048],[139.6326052,35.4487002],[139.6326628,35.4486519],[139.6328248,35.4484954],[139.6330258,35.4482747],[139.6331231,35.4481532],[139.6333844,35.4478537],[139.6334421,35.4478886],[139.6337658,35.4480774],[139.6340337,35.4478382],[139.6341066,35.4477569],[139.634235,35.4476283],[139.6347521,35.4471283],[139.6349523,35.4469976],[139.6351568,35.4468861],[139.6353202,35.4467969],[139.6353657,35.4467721],[139.635426,35.4467424],[139.6361448,35.4463484],[139.6362003,35.4463178],[139.6362821,35.4462719],[139.6363543,35.4462304],[139.6371917,35.4457669],[139.6382104,35.4452017],[139.6382538,35.445176],[139.638301,35.4451489],[139.6392859,35.444601],[139.6393434,35.4445699],[139.6395276,35.4447903],[139.6397238,35.4450464],[139.6399405,35.4453167],[139.6399985,35.4453845],[139.6400755,35.4454858],[139.6401389,35.4454536],[139.6400576,35.4453506],[139.639087,35.4441334]]}})

def test_cross_check():
    response = test_client.post("crosscheck", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[139.63920292723924,35.44528477213635],[139.63984460501422,35.44269282578362],[139.63774867207144,35.44401970338626],[139.6383402334955,35.44299558554904],[139.64009026937543,35.443326919216034],[139.64195818571795,35.442724984676666],[139.64219422011126,35.44454732194956],[139.64042302267546,35.444260670394954],[139.64041069847997,35.44528477213635],[139.63920292723924,35.44528477213635]]]},"properties":{}}]})
    assert response.status_code == 200
    assert response.json() == {"result":True}


@fixture
def current_client():
    yield TestClient(app)


def test_execute_kmeans_clustering_balance(current_client):
    """
    10個の拠点を3人で分割した場合、各矩形に3〜4個の拠点が
    含まれているか（均等性のテスト）を確認します。
    """
    # テスト用の入力データ
    test_data = {
        "k": 3,
        "points": [
            {"lat": 35.6, "lng": 139.7}, {"lat": 35.61, "lng": 139.71},
            {"lat": 35.62, "lng": 139.72}, {"lat": 35.63, "lng": 139.73},
            {"lat": 35.64, "lng": 139.74}, {"lat": 35.65, "lng": 139.75},
            {"lat": 35.66, "lng": 139.76}, {"lat": 35.67, "lng": 139.77},
            {"lat": 35.68, "lng": 139.78}, {"lat": 35.69, "lng": 139.79}
        ]
    }

    # API呼び出し
    response = current_client.post("/cluster/", json=test_data)
    
    # レスポンスのステータスコード確認
    assert response.status_code == 200
    
    data = response.json()
    rectangles = data["rectangles"]
    
    # 指定した人数分（k個）の矩形が返ってきているか
    assert len(rectangles) == 3
    
    # 各矩形の拠点数が期待通り（3個か4個）かを確認
    counts = [rect["count"] for rect in rectangles]
    for count in counts:
        assert count in [3, 4]
    
    # 全拠点の合計が一致するか
    assert sum(counts) == 10


def test_execute_kmeans_clustering_structure(current_client):
    """
    レスポンスのデータ構造（緯度経度の形式など）が正しいかを確認します。
    """
    test_data = {
        "k": 1,
        "points": [{"lat": 35.0, "lng": 135.0}, {"lat": 36.0, "lng": 136.0}]
    }
    response = current_client.post("/cluster/", json=test_data)
    rect = response.json()["rectangles"][0]
    
    # boundsが [[min_lat, min_lng], [max_lat, max_lng]] の形式か
    assert len(rect["bounds"]) == 2
    assert rect["bounds"][0][0] == 35.0  # min_lat
    assert rect["bounds"][1][0] == 36.0  # max_lat


def test_execute_kmeans_clustering_insufficient_points(current_client):
    """
    拠点数よりも担当者数(k)が多い場合に、400エラーが返るか確認します。
    """
    test_data = {
        "k": 5,
        "points": [
            {"lat": 35.0, "lng": 135.0},
            {"lat": 35.1, "lng": 135.1}
        ] # 2拠点しかないのに5分割しようとする
    }
    
    response = current_client.post("/cluster/", json=test_data)
    
    # 400エラーが返ってくることを期待
    assert response.status_code == 400


def test_execute_kmeans_clustering_invalid_k(current_client):
    """
    kが0以下の場合に、400エラーが返るか確認します。
    """
    test_data = {
        "k": 0,
        "points": [{"lat": 35.0, "lng": 135.0}]
    }
    response = current_client.post("/cluster/", json=test_data)
    assert response.status_code == 400