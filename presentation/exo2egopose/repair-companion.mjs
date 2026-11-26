import fs from 'node:fs/promises';import {meta,root} from './build-clean.mjs';
for(const i of [0,6,12,18]){let a=JSON.parse(await fs.readFile(root+`/batch-${i}.json`));for(const d of a){d.source=meta[d.index][4];d.el=d.el.map(e=>e.type==='text'&&e.v==='undefined'?{...e,v:d.source}:e);}await fs.writeFile(root+`/batch-${i}.json`,JSON.stringify(a));}
