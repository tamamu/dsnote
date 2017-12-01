

(defparameter +train-path+ "train.csv")

(defstruct (dataframe (:conc-name df-))
  (label nil)
  (datum #() :type array))

(defun df-index (df label)
  (typecase label
    (number label)
    (string (position label (df-label df) :test #'equal))))

(defun df-column (df label)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (res (make-array num-of-instances))
         (index (df-index df label)))
    (when index
      (loop for i from 0 below num-of-instances do
            (setf (aref res i) (aref (aref datum i) index)))
      res)))

(defun df-string-to-integer (df label)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (index (df-index df label)))
    (loop for i from 0 below num-of-instances do
          (setf (aref (aref datum i) index)
                (parse-integer (aref (aref datum i) index) :junk-allowed t)))))

(defun df-string-to-float (df label)
  (let* ((datum (df-datum df))
         (num-of-instances (length datum))
         (index (df-index df label)))
    (loop for i from 0 below num-of-instances do
          (setf (aref (aref datum i) index)
                (read-from-string (aref (aref datum i) index) nil)))))

(defun df-type-conversion (df types)
  (loop for cast in types
        for index = 0 then (1+ index) do
        (case cast
          ((integer number int)
           (df-string-to-integer df index))
          (float
           (df-string-to-float df index))
          (string nil))))

(defun counts (arr)
  (let ((ht (make-hash-table :test #'equal)))
    (loop for v across arr do
          (setf (gethash v ht)
                (1+ (gethash v ht 0))))
    (let* ((appears (make-hash-table :test #'equal))
           (diff 0)
           (counts (loop for key being each hash-key of ht
                        using (hash-value value)
                        unless (gethash key appears)
                        do (setf (gethash key appears) t
                                 diff (1+ diff))
                        collect (cons key value))))
      (values counts diff))))

(defun df-counts (df)
  (let ((label (df-label df)))
    (if label
        (loop for index across label do
              (multiple-value-bind (counts diff)
                (counts (df-column df index))
                (sort counts #'> :key #'cdr)
                (format t "~A~16T:~16T~A~16T~16T~A ~~~16T~A~%"
                        index diff (first counts) (car (last counts))))))))

(defun split (str delim)
  (let ((res (make-array 0 :element-type 'string
                           :fill-pointer 0
                           :adjustable t))
        (dlen (length delim)))
    (loop for i from 0 below (length str)
          for ch = (char str i)
          with dc = 0
          with word-start = 0
          with delim-start = -1
          with lq = nil
          with q = nil
          do
          (if (null q)
              (if (or (eq ch #\")
                      (eq ch #\'))
                  (setf q t
                        lq ch)
                  (if (eq ch (char delim dc))
                      (if (<= 0 delim-start)
                          (progn
                            (incf dc)
                            (when (= dc dlen)
                              (vector-push-extend (subseq str word-start delim-start) res)
                              (setf dc 0
                                    word-start (1+ i)
                                    delim-start -1)))
                          (progn
                            (setf delim-start i
                                  dc (1+ dc))
                            (when (= dc dlen)
                              (vector-push-extend (subseq str word-start delim-start) res)
                              (setf dc 0
                                    word-start (1+ i)
                                    delim-start -1))))
                      (when (< 0 dc)
                        (setf delim-start -1
                              dc 0))))
              (if (eq ch lq)
                  (setf q nil)))
          finally
          (if (< word-start i)
              (vector-push-extend (subseq str word-start i) res)))
    res))

(defun loadtxt (path delimiter)
  (let ((datum (make-array 0 :element-type 'array
                             :fill-pointer 0
                             :adjustable t)))
    (with-open-file (in path :direction :input)
      (loop for line = (read-line in nil nil) while line do
            (vector-push-extend (split line delimiter) datum)))
    datum))

(defun loadcsv (path &key header)
  (let ((data (loadtxt path ",")))
    (if header
        (make-dataframe :label (aref data 0)
                        :datum (subseq data 1))
        (make-dataframe :datum data))))

(let ((df (loadcsv +train-path+ :header t)))
  (df-type-conversion df '(int int int string string int int int string float string string))
  (terpri)
  (df-counts df)
)

