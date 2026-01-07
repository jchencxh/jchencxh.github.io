# Compatibility shim: Ruby 3.2+ removed taint/untaint/tainted?.
# Older Liquid/Jekyll versions still call `tainted?`, so we provide
# no-op implementations to keep local builds working.
unless "".respond_to?(:tainted?)
  class Object
    def tainted?; false; end
    def taint; self; end
    def untaint; self; end
  end
end
