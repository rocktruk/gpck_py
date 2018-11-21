# -*- coding:utf-8 -*-

import time
from http.server import BaseHTTPRequestHandler, HTTPServer,HTTPStatus
from urllib import parse
import base64
import json
import logging as log
import argparse
import uuid
from drawcard.drawPredict import Predict

HOST_NAME = '127.0.0.1'
PORT_NUMBER = 8080
CONIFRM_PATH = '/tmp'

class HttpHandler(BaseHTTPRequestHandler):
    def _set_headers(self,code=HTTPStatus.OK,length=0):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', length)
        self.send_header('Server', 'BaseServer-Python3.6')
        self.send_header('Date', self.date_time_string())
        self.end_headers()

    def _json_encode(self, data):
        array = data.split('&')
        json_data = {}
        for item in array:
            item = item.split('=', 1)
            json_data[item[0]] = item[1]
        return json_data

    def _get_handler(self, data):
        json_data = self._json_encode(data)

    def _post_handler(self, data):
        retVal = {}
        json_data = self._json_encode(data)
        file_name = json_data['FileName']
        file_data = base64.b64decode(json_data['FileData'])
        file_path = "%s/%s"% (CONIFRM_PATH, file_name)
        fd = open(file_path, 'w')
        fd.write(file_data)
        fd.close()
        retVal["RetCode"] = 0
        return json.dumps(retVal)

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        # self._set_headers()
        #get request params
        path = self.path
        query = parse.splitquery(path)
        self._get_handler(query[1]);

    def do_POST(self):
        msg = self.rfile.read(int(self.headers['content-length']))
        self.do_action(self.path,msg)


    def do_action(self,path,message):
        requestId = uuid.uuid4()
        log.info('%s|%s receive request msg %s'%(requestId,path,message))
        if path[1:].__eq__('quickdraw'):
            # p = Predict()
            reqmsg = json.loads(message)
            predict = FLAGS.p.draw_predict_fn(reqmsg['ink'],requestId)
            log.info('%s|response: %s'%(requestId,predict))
            self._set_headers(HTTPStatus.OK,len(predict))
            self.wfile.write(predict.encode('utf-8'))
        else:
            self._set_headers(HTTPStatus.NOT_FOUND)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes_file",
        type=str,
        default="rnn/eval.tfrecord.classes",
        help="Path to a file with the classes - one class per line")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="rnn",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--predict_data",
        type=str,
        default="predict/predict.tfrecord",
        help="Path to evaluation data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="logging level")
    parser.add_argument(
        "--succ_logit",
        type=float,
        default=80.0,
        help="识别成功分数，大于此分数直接返回对于标志，否则返回前4种类别")
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.p = Predict()
    FLAGS.p.FLAGS = FLAGS
    log.basicConfig(filename='/app/quickdraw/trace.log', level=FLAGS.log_level)
    httpd = HTTPServer((HOST_NAME, PORT_NUMBER), HttpHandler)
    print(time.asctime(), "Server Starts - %s:%s" % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, PORT_NUMBER))