import assert from 'node:assert/strict';
import {ProjectScope} from '../../monitoring/hosted-projects.js';

const client=projects=>({from(table){assert.equal(table,'monitor_projects');return {select(fields){assert.equal(fields,'id,name');return {order:async()=>({data:projects})}}}}});
const scope=new ProjectScope('emiel');
scope.reset('niels-user');
await scope.discover(client([{id:'niels',name:'Niels'}]));
assert.equal(scope.capture().projectId,'niels','a private member must not use the public default project');
assert.throws(()=>scope.choose('emiel'),/access required/);
const niels=scope.capture();
scope.reset('emiel-user');
assert.throws(()=>scope.assert(niels),/response discarded/);
await scope.discover(client([{id:'niels',name:'Niels'},{id:'emiel',name:'Emiel'}]));
assert.equal(scope.capture().projectId,'emiel');
const beforeSwitch=scope.capture();
scope.choose('niels');
assert.throws(()=>scope.assert(beforeSwitch),/response discarded/);
scope.reset('uninvited');await scope.discover(client([]));
assert.throws(()=>scope.capture(),/no project access/);
scope.reset();assert.throws(()=>scope.capture(),/Sign in/);

scope.reset('old-user');
let resolve;
const pending=scope.discover({from:()=>({select:()=>({order:()=>new Promise(r=>resolve=r)})})});
scope.reset('new-user');await scope.discover(client([{id:'new',name:'New'}]));
resolve({data:[{id:'old',name:'Old'}]});
assert.equal(await pending,false);
assert.equal(scope.capture().projectId,'new','late account discovery must not replace the current account');
console.log('Private project discovery, no-access, switching and stale-response checks passed.');
