import Promise from 'bluebird';
import username from 'username';
import config from './lib/config';
import services from './lib/services';
import proc from './lib/process';
import output from './lib/output';
import parsers from './lib/parsers';
import forwarders from './lib/forwarders';
import url from 'url';

const startForwarders = () => {
    const ports = parsers.parseForwardedPorts(manifest);
    // Pass; nothing to do
    if (!ports || !ports.length || !process.env.DOCKER_HOST) return Promise.resolve();
    const host = url.parse(process.env.DOCKER_HOST).hostname;
    if (!host) return Promise.reject(new Error('DOCKER_HOST is malformed. Cannot start forwarders.'));
    return forwarders.startForwarders(host, ports);
};