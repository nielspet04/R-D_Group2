import {validateEvent} from './contracts.ts';
import fixtures from '../../../tests/fixtures/monitoring/api-contracts.json' with {type:'json'};
Deno.test('valid shared Python/TypeScript event fixtures',()=>{for(const e of fixtures.valid)validateEvent(e)});
Deno.test('invalid shared Python/TypeScript event fixtures',()=>{for(const e of fixtures.invalid){let threw=false;try{validateEvent(e)}catch{threw=true}if(!threw)throw Error('Invalid fixture accepted: '+JSON.stringify(e))}});
