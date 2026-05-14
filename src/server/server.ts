import express from 'express';
import bodyParser from 'body-parser';
import config from 'config';
import moment from 'moment';
import crypto from 'crypto';
import cors from 'cors';
import qs from 'qs';

const app = express();

const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

// const uri = 'mongodb://localhost:27017';
// const dbName = 'Wildlife_Conservation_Website';

// const client = new MongoClient(uri);

// async function connectMongo() {
//     try {
//         await client.connect();
//         console.log('Connected to MongoDB');
//         const db = client.db(dbName);
//     } catch (err) {
//         console.error('Error connecting to MongoDB:', err);
//     }
// }
// connectMongo();

app.post('/api/payment/create_payment_url', (req, res) => {
    try {
        process.env.TZ = 'Asia/Ho_Chi_Minh';
        const date = new Date();
        const createDate = moment(date).format('YYYYMMDDHHmmss');
        const ipAddr = req.headers['x-forwarded-for'] || req.socket.remoteAddress || req.connection.remoteAddress;

        console.log('Config values:');
        console.log('vnp_TmnCode:', config.get('vnp_TmnCode'));
        console.log('vnp_HashSecret:', config.get('vnp_HashSecret'));
        console.log('vnp_Url:', config.get('vnp_Url'));
        console.log('vnp_ReturnUrl:', config.get('vnp_ReturnUrl'));

        const tmnCode: string = config.get('vnp_TmnCode') as string;
        const secretKey: string = config.get('vnp_HashSecret');
        let vnpUrl: string = config.get('vnp_Url');
        const returnUrl: string = config.get('vnp_ReturnUrl');
        // Lưu projectId
        const projectId = req.body.projectId;
        const orderId = `${moment(date).format('DDHHmmss')}_${projectId}`;

        const amount = req.body.amount;
        const bankCode = req.body.bankCode || '';
        const locale = req.body.locale || 'vn';
        const currCode = 'VND';

        let vnp_Params: any = {};
        vnp_Params['vnp_Version'] = '2.1.0';
        vnp_Params['vnp_Command'] = 'pay';
        vnp_Params['vnp_TmnCode'] = tmnCode;
        vnp_Params['vnp_Locale'] = locale;
        vnp_Params['vnp_CurrCode'] = currCode;
        vnp_Params['vnp_TxnRef'] = orderId;
        vnp_Params['vnp_OrderInfo'] = 'Thanh toan cho ma GD:' + orderId;
        vnp_Params['vnp_OrderType'] = 'other';
        vnp_Params['vnp_Amount'] = amount * 100;
        vnp_Params['vnp_ReturnUrl'] = returnUrl;
        vnp_Params['vnp_IpAddr'] = ipAddr;
        vnp_Params['vnp_CreateDate'] = createDate;

        if (bankCode) {
            vnp_Params['vnp_BankCode'] = bankCode;
        }

        vnp_Params = sortObject(vnp_Params);

        const signData = qs.stringify(vnp_Params, { encode: false });
        const hmac = crypto.createHmac('sha512', secretKey as crypto.BinaryLike);
        const signed = hmac.update(Buffer.from(signData, 'utf-8')).digest('hex');
        vnp_Params['vnp_SecureHash'] = signed;
        vnpUrl += '?' + qs.stringify(vnp_Params, { encode: false });

        console.log('Generated vnpUrl:', vnpUrl);

        res.json({ redirectUrl: vnpUrl });

        
    } catch (error) {
        console.error('Error creating payment URL:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

app.get('/vnpay_return', async (req, res) => {
    try {
        let vnp_Params = req.query;

        const secureHash = vnp_Params['vnp_SecureHash'];

        delete vnp_Params['vnp_SecureHash'];
        delete vnp_Params['vnp_SecureHashType'];

        vnp_Params = sortObject(vnp_Params);

        const secretKey: string = config.get('vnp_HashSecret');
        const signData = qs.stringify(vnp_Params, { encode: false });
        const hmac = crypto.createHmac('sha512', secretKey as crypto.BinaryLike);
        const signed = hmac.update(Buffer.from(signData, 'utf-8')).digest('hex');

        if (secureHash === signed) {
            res.json({ message: 'Giao dịch thành công', data: vnp_Params });
        } else {
            res.json({ message: 'Giao dịch không hợp lệ' });
        }
    } catch (error) {
        console.error('Error processing vnpay return:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

function sortObject(obj: any) {
    const sorted: any = {};
    const keys = Object.keys(obj).sort();
    keys.forEach(key => {
        sorted[key] = encodeURIComponent(obj[key]).replace(/%20/g, '+');
    });
    return sorted;
}

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
