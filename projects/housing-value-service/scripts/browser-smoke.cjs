// Run by CI against the running Docker container, using Playwright.
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true, channel: 'chrome'});
  const page = await browser.newPage({viewport: {width: 1440, height: 1120}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:8000', {waitUntil: 'networkidle'});
  await page.getByRole('button', {name: 'Estimate district value'}).click();
  await page.waitForFunction(() => document.getElementById('prediction').textContent.startsWith('$'));
  const response = JSON.parse(await page.locator('#response').textContent());
  if (!response.trace_first_record) throw new Error('Missing prediction trace');
  if (await page.locator('#error').textContent()) throw new Error('Unexpected UI error');
  await page.screenshot({path: 'reports/demo-desktop.png', fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: 'reports/demo-mobile.png', fullPage: true});
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  if (overflow) throw new Error('Mobile horizontal overflow');
  if (errors.length) throw new Error(errors.join('; '));
  console.log(JSON.stringify({browser_prediction: response.predictions[0].value,
    trace: true, javascript_errors: errors, mobile_overflow: overflow}));
  await browser.close();
})().catch(error => {console.error(error); process.exit(1);});
