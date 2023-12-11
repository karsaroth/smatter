const express = require('express');
const ws = require('ws');
const net = require('net');
const http = require('http');
const logger = require('pino')();

/**
 * Used to handle errors on the backend socket
 * Returned function is used as a callback for the error (TCP socket) event
 * @param {ws.WebSocket} socket
 * @returns {(Error) => void}
 */
function backendErrorWithWs(socket) {
  function backendError(err) {
    logger.info('Error on backend socket.');
    logger.error(err);
    socket.close();
  }

  return backendError;
}

/**
 * Used to handle data on the backend socket
 * Returned function is used as a callback for the data (TCP socket read) event
 * @param {ws.WebSocket} socket
 * @returns {(Buffer) => void}
 */
function backendDataWithWs(socket) {
  function backendData(
    /** @type {Buffer} */
    data
  ) {
    const messages = bufferBackendData(socket, data);
    for (const message of messages) {
      logger.info(`Forwarding from transx: ${message}`)
      socket.send(message);
    }
  }

  return backendData;
}

/**
 * Ensure we read a complete message from the backend socket
 * @param {net.Socket} socket 
 * @param {Buffer} data 
 * @returns {string[]}
 */
function bufferBackendData(socket, data){
  const messages = [];
  /** @type {Buffer} */
  let chunk = socket.chunk;

  if (!chunk) {
    chunk = Buffer.alloc(0);
  }

  //store the incoming data
  chunk = Buffer.concat([chunk, data]);

  //Check if the buffer contains a newline character (\n)
  let newline = chunk.indexOf('\n', 0, 'utf8');
  while (newline > -1) {
    //if there is a new line, then you have a complete message
    var message = chunk.subarray(0, newline);
    messages.push(message.toString('utf8'));
    chunk = Buffer.from(chunk.subarray(newline + 1));
    newline = chunk.indexOf('\n', 0, 'utf8');
  }
  socket.chunck = chunk;
  return messages;
}



/**
 * Used to handle messages on an open websocket
 * Returned function is used as a callback for the message (ws inbound) event
 * @param {net.Socket} backendSocket 
 * @returns {(string) => void}
 */
function messageWithSocket(
  backendSocket
) {
  function message(message) {
    backendSocket.write(message);
  }

  return message;
}

/**
 * Used to end a connection on the websocket
 * Returned function is used as a callback for the close event
 * @param {net.Socket} backendSocket 
 * @returns {() => void}
 */
function connectionCloseWithSocket(
  backendSocket
) {
  function close() {
    logger.info('Closing connection on backend socket because websocket is closing.');
    backendSocket.end();
  }

  return close;
}



/**
 * Used to handle a new connection on the websocket
 * @param {ws.WebSocket} socket
 * @param {http.IncomingMessage} _request
 */
function connection(
  socket,
  _request
) {
  logger.info('New connection on backend socket.');
  const backendSocket = net.createConnection(process.env.PY_TCP_PORT || 9999, 'localhost');
  backendSocket.on('error', backendErrorWithWs(socket));
  backendSocket.on('data', backendDataWithWs(socket));

  socket.on('message', messageWithSocket(backendSocket));
  socket.on('close', connectionCloseWithSocket(backendSocket));
  socket.on('error', (err) => {
    logger.info('Error on websocket.');
    logger.error(err);
    socket.close();
  });
}


/**
 * Used to emit a connection event on the websocket
 * Returned function is used as a callback for handleUpgrade event
 * @param {ws.Server} ws
 * @param {http.IncomingMessage} request
 * @returns {(net.Socket) => void}
 */
function emitConnectionWithRequest(ws, request) {
  function emitConnection(s) {
    logger.info('Emitting connection event on websocket server.');
    ws.emit('connection', s, request);
  }
  return emitConnection;
}

/**
 * Used to handle an upgrade request on the websocket
 * Returned function is used as a callback for the upgrade event
 * @param {ws.Server} ws
 * @returns {(http.IncomingMessage, net.Socket, Buffer) => void}
 */
function upgradeWithServer(ws) {
  function upgrade(r, s, h) {
    logger.info('Upgrade request received, connecting with headless websocket server.');
    return ws.handleUpgrade(r, s, h, emitConnectionWithRequest(ws, r));
  }
  return upgrade;
}

/**
 * Main function preparing and starting express/ws server.
 */
function main() {
  const app = express();
  const wsHeadless = new ws.Server(
    { noServer: true }, 
    () => logger.info('Headless websocket server started.')
  );
  
  wsHeadless.on('connection', connection);

  if (process.env.ENABLE_TEST_ENDPOINT && process.env.ENABLE_TEST_ENDPOINT === 'true') {
    app.use(express.static('public'));
  }

  const server = app.listen(process.env.WS_PORT || 9998);
  server.on('upgrade', upgradeWithServer(wsHeadless));
  logger.info('Server has started.');
}

main();