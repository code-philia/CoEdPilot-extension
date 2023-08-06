var server = http.createServer();

var HttpTests = {
  testCreateServerReturnsServer: function() {
    vassert.assertTrue(server instanceof http.Server);
    vassert.testComplete();
  },

  testServerListeningEvent: function() {
    server.listen(test_options.port, function() {
      vassert.testComplete();
    });
  },

  // More ...
}