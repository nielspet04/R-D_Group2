import {chromium} from '/Users/nielspeters/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core/index.mjs';
import fs from 'node:fs/promises';import {pathToFileURL} from 'node:url';import {root} from './build-clean.mjs';
const browser=await chromium.launch({executablePath:'/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',headless:true});const page=await browser.newPage({viewport:{width:1280,height:720},reducedMotion:'reduce'});await fs.mkdir(root+'/preview',{recursive:true});
await page.goto(pathToFileURL(root+'/Exo2EgoPose-animated.html').href);const errors=[];page.on('pageerror',e=>errors.push(e.message));
for(let i=0;i<24;i++){await page.evaluate(i=>show(i),i);await page.screenshot({path:root+`/preview/slide-${i+1}.png`});}
await page.keyboard.press('Home');await page.keyboard.press('ArrowRight');const navigation=await page.evaluate(()=>document.querySelector('.slide.active').dataset.index);await page.keyboard.press('n');const notes=await page.locator('#notes').evaluate(x=>x.classList.contains('open'));await page.keyboard.press('n');
await page.pdf({path:root+'/Exo2EgoPose-Toledo.pdf',printBackground:true,preferCSSPageSize:true});
console.log(JSON.stringify({navigation,notes,errors,pdf:root+'/Exo2EgoPose-Toledo.pdf'}));await browser.close();
